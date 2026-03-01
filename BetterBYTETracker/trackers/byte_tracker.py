"""Fully vectorized SoA BYTETracker — no per-track Python objects."""

from __future__ import annotations

import numpy as np

from ..utils.ops import xywh2ltwh, batch_tlwh_to_xyah, batch_means_to_xyxy
from .basetrack import NEW, TRACKED, LOST, REMOVED
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class BYTETracker:
    """Fully vectorized BYTETracker using Struct-of-Arrays layout.

    All track state is stored in contiguous numpy arrays. No per-track Python objects.

    Input: (N, 6) array of [x, y, w, h, conf, cls] per detection.
    Output: (M, 8) array of [x1, y1, x2, y2, track_id, score, cls, idx].
    """
    
    def __init__(self, args, frame_rate: int = 30, max_tracks: int = 512):
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilterXYAH()
        self._next_id = 0
        self.max_tracks = max_tracks
        self._high = 0  # high-water mark: all live tracks are in [:_high]

        # Pre-allocated SoA track state — REMOVED slots are free
        self.means = np.zeros((max_tracks, 8), dtype=np.float32)
        self.covariances = np.zeros((max_tracks, 8, 8), dtype=np.float32)
        self.scores = np.zeros(max_tracks, dtype=np.float32)
        self.cls = np.zeros(max_tracks, dtype=np.float32)
        self.track_ids = np.zeros(max_tracks, dtype=np.int32)
        self.states = np.full(max_tracks, REMOVED, dtype=np.int8)
        self.is_activated = np.zeros(max_tracks, dtype=np.bool_)
        self.frame_ids = np.zeros(max_tracks, dtype=np.int32)
        self.start_frames = np.zeros(max_tracks, dtype=np.int32)
        self.tracklet_lens = np.zeros(max_tracks, dtype=np.int32)
        self.idx = np.zeros(max_tracks, dtype=np.float32)

    def _alloc_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_id + 1, self._next_id + 1 + n, dtype=np.int32)
        self._next_id += n
        return ids

    @staticmethod
    def _det_to_xyah(xywh: np.ndarray) -> np.ndarray:
        return batch_tlwh_to_xyah(xywh2ltwh(xywh))

    def predict(self):
        """Run Kalman prediction on all active tracks, updating means (N, 8) and covariances (N, 8, 8) in-place.

        Tracks in LOST state have their height velocity zeroed before prediction.
        Only tracks with state TRACKED or LOST are predicted.
        """
        predict_mask = (self.states[:self._high] == TRACKED) | (self.states[:self._high] == LOST)
        predict_idx = np.where(predict_mask)[0]
        if len(predict_idx) == 0:
            return

        pred_means = self.means[predict_idx].copy()
        pred_covs = self.covariances[predict_idx]

        not_tracked = self.states[predict_idx] != TRACKED
        pred_means[not_tracked, 7] = 0

        pred_means, pred_covs = self.kalman_filter.batch_predict(pred_means, pred_covs)
        self.means[predict_idx] = pred_means
        self.covariances[predict_idx] = pred_covs

    def multi_predict(self, tracks):
        """Alias for predict(), kept for API parity with ultralytics BYTETracker. tracks arg does not do anything"""
        self.predict()

    def get_active_tracks(self) -> np.ndarray:
        """Return currently active (tracked and activated) tracks.

        Returns:
            (M, 8) array of [x1, y1, x2, y2, track_id, score, cls, idx].
            Empty (0, 8) array if no active tracks.
        """
        h = self._high
        out_mask = (self.states[:h] == TRACKED) & self.is_activated[:h]
        if out_mask.any():
            out_xyxy = batch_means_to_xyxy(self.means[:h][out_mask])
            return np.concatenate([
                out_xyxy,
                self.track_ids[:h][out_mask].astype(np.float32).reshape(-1, 1),
                self.scores[:h][out_mask].reshape(-1, 1),
                self.cls[:h][out_mask].reshape(-1, 1),
                self.idx[:h][out_mask].reshape(-1, 1),
            ], axis=1)
        return np.empty((0, 8), dtype=np.float32)

    def get_active_tracks_with_lifetime(self) -> np.ndarray:
        """Return active tracks with frame lifetime info.

        Returns:
            (M, 10) array of [x1, y1, x2, y2, track_id, score, cls, idx, start_frame, last_frame].
            Empty (0, 10) array if no active tracks.
        """
        base = self.get_active_tracks()
        if len(base) == 0:
            return np.empty((0, 10), dtype=np.float32)
        h = self._high
        out_mask = (self.states[:h] == TRACKED) & self.is_activated[:h]
        return np.concatenate([
            base,
            self.start_frames[:h][out_mask].astype(np.float32).reshape(-1, 1),
            self.frame_ids[:h][out_mask].astype(np.float32).reshape(-1, 1),
        ], axis=1)

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    def _apply_match(self, m_idx, det_xyah, det_scores, det_cls, det_idx_vals, m_local, m_det):
        """Apply Kalman update and state changes for a set of matches."""
        old_lens = self.tracklet_lens[m_idx].copy()
        was_tracked = self.states[m_idx] == TRACKED

        # Batch Kalman update
        new_means, new_covs = self.kalman_filter.batch_update(
            self.means[m_idx], self.covariances[m_idx], det_xyah[m_det]
        )
        self.means[m_idx] = new_means
        self.covariances[m_idx] = new_covs

        self.scores[m_idx] = det_scores[m_det]
        self.cls[m_idx] = det_cls[m_det]
        self.idx[m_idx] = det_idx_vals[m_det]
        self.frame_ids[m_idx] = self.frame_id
        self.states[m_idx] = TRACKED
        self.is_activated[m_idx] = True
        # tracked: old_len + 1, lost (re-activate): 0
        self.tracklet_lens[m_idx] = np.where(was_tracked, old_lens + 1, np.int32(0))

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame of detections.

        Args:
            detections: (N, 6) array of [x, y, w, h, conf, cls].
                        Also accepts (6,) for a single detection.

        Returns:
            (M, 8) array of [x1, y1, x2, y2, track_id, score, cls, idx].
        """
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim == 1:
            detections = detections[np.newaxis]
        self.frame_id += 1

        # ---- 1. Split detections by confidence ----
        xywh = detections[..., :4]
        scores = detections[..., 4]
        cls_vals = detections[..., 5]

        high_mask = scores >= self.args.track_high_thresh
        low_mask = (scores > self.args.track_low_thresh) & ~high_mask

        n_high = int(high_mask.sum())
        n_low = int(low_mask.sum())

        if n_high > 0:
            high_xywh = xywh[high_mask]
            high_xyah = self._det_to_xyah(high_xywh)
            high_xyxy = self._xywh_to_xyxy(high_xywh)
            high_scores = scores[high_mask]
            high_cls = cls_vals[high_mask]
            high_idx_vals = np.where(high_mask)[0].astype(np.float32)
        else:
            high_xyah = np.empty((0, 4), dtype=np.float32)
            high_xyxy = np.empty((0, 4), dtype=np.float32)
            high_scores = np.empty(0, dtype=np.float32)
            high_cls = np.empty(0, dtype=np.float32)
            high_idx_vals = np.empty(0, dtype=np.float32)

        if n_low > 0:
            low_xywh = xywh[low_mask]
            low_xyah = self._det_to_xyah(low_xywh)
            low_xyxy = self._xywh_to_xyxy(low_xywh)
            low_scores = scores[low_mask]
            low_cls = cls_vals[low_mask]
            low_idx_vals = np.where(low_mask)[0].astype(np.float32)
        else:
            low_xyah = np.empty((0, 4), dtype=np.float32)
            low_xyxy = np.empty((0, 4), dtype=np.float32)
            low_scores = np.empty(0, dtype=np.float32)
            low_cls = np.empty(0, dtype=np.float32)
            low_idx_vals = np.empty(0, dtype=np.float32)

        # ---- 2. Build track masks ----
        h = self._high
        states_h = self.states[:h]
        tracked_activated_mask = (states_h == TRACKED) & self.is_activated[:h]
        unconfirmed_mask = (states_h == TRACKED) & ~self.is_activated[:h]
        lost_mask = states_h == LOST
        pool_mask = tracked_activated_mask | lost_mask

        pool_idx = np.where(pool_mask)[0]
        unconf_idx = np.where(unconfirmed_mask)[0]

        # ---- 3. Kalman predict pool + unconfirmed ----
        self.predict()

        # ---- 4. First association: pool tracks vs high-confidence dets ----
        u_detection_high = np.arange(n_high)
        u_pool = np.arange(len(pool_idx))

        if len(pool_idx) > 0 and n_high > 0:
            pool_xyxy = batch_means_to_xyxy(self.means[pool_idx])
            dists = matching.iou_distance(pool_xyxy, high_xyxy)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, high_scores)

            matches, u_pool, u_detection_high = matching.linear_assignment(dists, thresh=self.args.match_thresh)

            if len(matches) > 0:
                matches = np.asarray(matches)
                m_idx = pool_idx[matches[:, 0]]
                self._apply_match(m_idx, high_xyah, high_scores, high_cls, high_idx_vals, matches[:, 0], matches[:, 1])

        # ---- 5. Second association: remaining tracked vs low-confidence dets ----
        if len(u_pool) > 0:
            u_pool_global = pool_idx[u_pool]
            r_tracked_mask = self.states[u_pool_global] == TRACKED
            r_tracked_idx = u_pool_global[r_tracked_mask]
        else:
            r_tracked_idx = np.empty(0, dtype=np.int64)

        u_r_tracked = np.arange(len(r_tracked_idx))

        if len(r_tracked_idx) > 0 and n_low > 0:
            rt_xyxy = batch_means_to_xyxy(self.means[r_tracked_idx])
            dists = matching.iou_distance(rt_xyxy, low_xyxy)
            matches, u_r_tracked, _ = matching.linear_assignment(dists, thresh=0.5)

            if len(matches) > 0:
                matches = np.asarray(matches)
                m_idx = r_tracked_idx[matches[:, 0]]
                self._apply_match(m_idx, low_xyah, low_scores, low_cls, low_idx_vals, matches[:, 0], matches[:, 1])

        # Mark unmatched remaining tracked as lost
        if len(u_r_tracked) > 0:
            u_rt_idx = r_tracked_idx[u_r_tracked]
            still_tracked = self.states[u_rt_idx] != LOST
            to_lose = u_rt_idx[still_tracked]
            self.states[to_lose] = LOST

        # ---- 6. Third association: unconfirmed tracks vs remaining high dets ----
        remaining_high_det = u_detection_high

        if len(unconf_idx) > 0 and len(remaining_high_det) > 0:
            uc_xyxy = batch_means_to_xyxy(self.means[unconf_idx])
            rem_xyxy = high_xyxy[remaining_high_det]

            dists = matching.iou_distance(uc_xyxy, rem_xyxy)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, high_scores[remaining_high_det])

            matches, u_unconfirmed, u_det_rem = matching.linear_assignment(dists, thresh=0.7)

            if len(matches) > 0:
                matches = np.asarray(matches)
                m_det_global = np.asarray(remaining_high_det)[matches[:, 1]]
                m_idx = unconf_idx[matches[:, 0]]

                new_means, new_covs = self.kalman_filter.batch_update(
                    self.means[m_idx], self.covariances[m_idx], high_xyah[m_det_global]
                )
                self.means[m_idx] = new_means
                self.covariances[m_idx] = new_covs
                self.scores[m_idx] = high_scores[m_det_global]
                self.cls[m_idx] = high_cls[m_det_global]
                self.idx[m_idx] = high_idx_vals[m_det_global]
                self.states[m_idx] = TRACKED
                self.is_activated[m_idx] = True
                self.frame_ids[m_idx] = self.frame_id
                self.tracklet_lens[m_idx] += 1

            # Mark unmatched unconfirmed as removed
            if len(u_unconfirmed) > 0:
                self.states[unconf_idx[u_unconfirmed]] = REMOVED

            remaining_high_det = np.asarray(remaining_high_det)[np.asarray(u_det_rem)] if len(u_det_rem) > 0 else np.array([], dtype=int)
        elif len(unconf_idx) > 0:
            self.states[unconf_idx] = REMOVED

        # ---- 7. Init new tracks ----
        if len(remaining_high_det) > 0:
            new_scores = high_scores[remaining_high_det]
            new_mask = new_scores >= self.args.new_track_thresh
            new_det = remaining_high_det[new_mask]

            if len(new_det) > 0:
                free_slots = np.where(self.states[:self._high] == REMOVED)[0]
                if len(free_slots) < len(new_det):
                    # Extend into unused buffer space
                    extra = min(len(new_det) - len(free_slots), self.max_tracks - self._high)
                    if extra > 0:
                        extra_slots = np.arange(self._high, self._high + extra)
                        free_slots = np.concatenate([free_slots, extra_slots])
                n_new = min(len(new_det), len(free_slots))
                if n_new > 0:
                    new_det = new_det[:n_new]
                    slots = free_slots[:n_new]
                    new_xyah = high_xyah[new_det]
                    new_means, new_covs = self.kalman_filter.batch_initiate(new_xyah)
                    new_ids = self._alloc_ids(n_new)

                    self.means[slots] = new_means
                    self.covariances[slots] = new_covs
                    self.scores[slots] = high_scores[new_det]
                    self.cls[slots] = high_cls[new_det]
                    self.track_ids[slots] = new_ids
                    self.states[slots] = TRACKED
                    self.is_activated[slots] = self.frame_id == 1
                    self.frame_ids[slots] = self.frame_id
                    self.start_frames[slots] = self.frame_id
                    self.tracklet_lens[slots] = 0
                    self.idx[slots] = high_idx_vals[new_det]
                    self._high = max(self._high, int(slots[-1]) + 1)

        # ---- 8. Expire lost tracks ----
        h = self._high
        lost_expire = (self.states[:h] == LOST) & ((self.frame_id - self.frame_ids[:h]) > self.max_time_lost)
        if lost_expire.any():
            expire_idx = np.where(lost_expire)[0]
            self.states[expire_idx] = REMOVED

        # ---- 9. Remove duplicates ----
        self._remove_duplicates()

        # ---- 10. Output ----
        return self.get_active_tracks()

    def _remove_duplicates(self):
        h = self._high
        tracked_mask = (self.states[:h] == TRACKED) & self.is_activated[:h]
        lost_mask = self.states[:h] == LOST
        tracked_idx = np.where(tracked_mask)[0]
        lost_idx = np.where(lost_mask)[0]

        if len(tracked_idx) == 0 or len(lost_idx) == 0:
            return

        pdist = matching.iou_distance(
            batch_means_to_xyxy(self.means[tracked_idx]),
            batch_means_to_xyxy(self.means[lost_idx]),
        )
        pairs = np.where(pdist < 0.15)
        if len(pairs[0]) == 0:
            return

        to_remove = []
        for p, q in zip(pairs[0], pairs[1]):
            timep = int(self.frame_ids[tracked_idx[p]]) - int(self.start_frames[tracked_idx[p]])
            timeq = int(self.frame_ids[lost_idx[q]]) - int(self.start_frames[lost_idx[q]])
            if timep > timeq:
                to_remove.append(int(lost_idx[q]))
            else:
                to_remove.append(int(tracked_idx[p]))

        if to_remove:
            self.states[np.asarray(to_remove, dtype=np.int64)] = REMOVED

    def reset(self):
        self.frame_id = 0
        self._next_id = 0
        self.states[:self._high] = REMOVED
        self._high = 0

    @staticmethod
    def reset_id():
        pass
