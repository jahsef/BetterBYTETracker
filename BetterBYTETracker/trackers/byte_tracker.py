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

    def __init__(self, args, frame_rate: int = 30):
        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = KalmanFilterXYAH()
        self._next_id = 0

        # SoA track state — all empty at start
        self.means = np.empty((0, 8), dtype=np.float32)
        self.covariances = np.empty((0, 8, 8), dtype=np.float32)
        self.scores = np.empty(0, dtype=np.float32)
        self.cls = np.empty(0, dtype=np.float32)
        self.track_ids = np.empty(0, dtype=np.int32)
        self.states = np.empty(0, dtype=np.int8)
        self.is_activated = np.empty(0, dtype=np.bool_)
        self.frame_ids = np.empty(0, dtype=np.int32)
        self.start_frames = np.empty(0, dtype=np.int32)
        self.tracklet_lens = np.empty(0, dtype=np.int32)
        self.idx = np.empty(0, dtype=np.float32)

    def _alloc_ids(self, n: int) -> np.ndarray:
        ids = np.arange(self._next_id + 1, self._next_id + 1 + n, dtype=np.int32)
        self._next_id += n
        return ids

    @staticmethod
    def _det_to_xyah(xywh: np.ndarray) -> np.ndarray:
        return batch_tlwh_to_xyah(xywh2ltwh(xywh))

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
        tracked_activated_mask = (self.states == TRACKED) & self.is_activated
        unconfirmed_mask = (self.states == TRACKED) & ~self.is_activated
        lost_mask = self.states == LOST
        pool_mask = tracked_activated_mask | lost_mask

        pool_idx = np.where(pool_mask)[0]
        unconf_idx = np.where(unconfirmed_mask)[0]

        # ---- 3. Kalman predict pool + unconfirmed ----
        predict_mask = pool_mask | unconfirmed_mask
        predict_idx = np.where(predict_mask)[0]
        if len(predict_idx) > 0:
            pred_means = self.means[predict_idx].copy()
            pred_covs = self.covariances[predict_idx]

            # Zero vh for non-tracked (lost) tracks
            not_tracked = self.states[predict_idx] != TRACKED
            pred_means[not_tracked, 7] = 0

            pred_means, pred_covs = self.kalman_filter.batch_predict(pred_means, pred_covs)
            self.means[predict_idx] = pred_means
            self.covariances[predict_idx] = pred_covs

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
                new_xyah = high_xyah[new_det]
                new_means, new_covs = self.kalman_filter.batch_initiate(new_xyah)
                n_new = len(new_det)
                new_ids = self._alloc_ids(n_new)

                self.means = np.concatenate([self.means, new_means])
                self.covariances = np.concatenate([self.covariances, new_covs])
                self.scores = np.concatenate([self.scores, high_scores[new_det]])
                self.cls = np.concatenate([self.cls, high_cls[new_det]])
                self.track_ids = np.concatenate([self.track_ids, new_ids])
                self.states = np.concatenate([self.states, np.full(n_new, TRACKED, dtype=np.int8)])
                self.is_activated = np.concatenate([self.is_activated, np.full(n_new, self.frame_id == 1, dtype=np.bool_)])
                self.frame_ids = np.concatenate([self.frame_ids, np.full(n_new, self.frame_id, dtype=np.int32)])
                self.start_frames = np.concatenate([self.start_frames, np.full(n_new, self.frame_id, dtype=np.int32)])
                self.tracklet_lens = np.concatenate([self.tracklet_lens, np.zeros(n_new, dtype=np.int32)])
                self.idx = np.concatenate([self.idx, high_idx_vals[new_det]])

        # ---- 8. Expire lost tracks ----
        lost_expire = (self.states == LOST) & ((self.frame_id - self.frame_ids) > self.max_time_lost)
        self.states[lost_expire] = REMOVED

        # ---- 9. Remove duplicates ----
        self._remove_duplicates()

        # ---- 10. Prune removed ----
        keep = self.states != REMOVED
        self._apply_mask(keep)

        # ---- 11. Output ----
        out_mask = (self.states == TRACKED) & self.is_activated
        n_out = int(out_mask.sum())
        if n_out > 0:
            out_means = self.means[out_mask]
            out_xyxy = batch_means_to_xyxy(out_means)
            return np.concatenate([
                out_xyxy,
                self.track_ids[out_mask].astype(np.float32).reshape(-1, 1),
                self.scores[out_mask].reshape(-1, 1),
                self.cls[out_mask].reshape(-1, 1),
                self.idx[out_mask].reshape(-1, 1),
            ], axis=1)
        return np.empty((0, 8), dtype=np.float32)

    def _apply_mask(self, mask: np.ndarray):
        self.means = self.means[mask]
        self.covariances = self.covariances[mask]
        self.scores = self.scores[mask]
        self.cls = self.cls[mask]
        self.track_ids = self.track_ids[mask]
        self.states = self.states[mask]
        self.is_activated = self.is_activated[mask]
        self.frame_ids = self.frame_ids[mask]
        self.start_frames = self.start_frames[mask]
        self.tracklet_lens = self.tracklet_lens[mask]
        self.idx = self.idx[mask]

    def _remove_duplicates(self):
        tracked_mask = (self.states == TRACKED) & self.is_activated
        lost_mask = self.states == LOST
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
        fr = int(30.0 * self.max_time_lost / self.args.track_buffer) if self.args.track_buffer > 0 else 30
        self.__init__(self.args, frame_rate=fr)

    @staticmethod
    def reset_id():
        pass
