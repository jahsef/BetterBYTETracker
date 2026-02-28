"""Compare BetterBYTETracker output against ultralytics BYTETracker on identical inputs.

Requires ultralytics to be installed (test-only dependency).
Run: python -m pytest BetterBYTETracker/tests/test_parity.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Ultralytics needs a results object with .xywh/.conf/.cls attributes
# ---------------------------------------------------------------------------

class _UltraResults:
    """Minimal results object for ultralytics BYTETracker.update()."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _UltraResults(self.xywh[idx], self.conf[idx], self.cls[idx])


# ---------------------------------------------------------------------------
# Deterministic detection sequence (5 frames, 3 objects with slight motion)
# ---------------------------------------------------------------------------

def _make_detections(seed: int = 42):
    """Generate a short deterministic detection sequence.

    Returns (our_frames, ultra_frames) where:
        our_frames: list of (N, 6) arrays [x, y, w, h, conf, cls]
        ultra_frames: list of _UltraResults for ultralytics
    """
    rng = np.random.RandomState(seed)
    n_frames, n_objects = 5, 3

    base_xywh = np.array(
        [[100, 100, 40, 60],
         [250, 200, 50, 80],
         [400, 150, 30, 50]],
        dtype=np.float32,
    )

    our_frames = []
    ultra_frames = []
    for f in range(n_frames):
        noise = rng.randn(n_objects, 4).astype(np.float32) * [2, 2, 0.5, 0.5]
        xywh = base_xywh + noise + np.array([f * 5, f * 3, 0, 0], dtype=np.float32)
        conf = np.clip(0.85 + rng.randn(n_objects).astype(np.float32) * 0.05, 0.5, 1.0)
        cls = np.zeros(n_objects, dtype=np.float32)
        our_frames.append(np.column_stack([xywh, conf, cls]))
        ultra_frames.append(_UltraResults(xywh, conf, cls))
        base_xywh[:, 0] += 3
        base_xywh[:, 1] += 2
    return our_frames, ultra_frames


# ---------------------------------------------------------------------------
# Default tracker args
# ---------------------------------------------------------------------------

TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ours(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Run our BYTETracker — takes (N, 6) arrays directly."""
    import importlib

    mod = importlib.import_module("BetterBYTETracker.trackers.byte_tracker")
    BYTETracker = mod.BYTETracker

    tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)
    outputs = []
    for det in frames:
        out = tracker.update(det)
        outputs.append(out.copy() if len(out) else np.empty((0, 8), dtype=np.float32))
    return outputs


def _run_ultra(frames: list[_UltraResults]) -> list[np.ndarray]:
    """Run the original ultralytics BYTETracker."""
    from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: F811

    tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)
    outputs = []
    for det in frames:
        out = tracker.update(det)
        outputs.append(out.copy() if len(out) else np.empty((0, 8), dtype=np.float32))
    return outputs


def _sort_by_track_id(arr: np.ndarray) -> np.ndarray:
    """Sort rows by track_id column (index 4) for stable comparison."""
    if len(arr) == 0:
        return arr
    return arr[arr[:, 4].argsort()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParity:
    """Verify our BYTETracker produces identical output to ultralytics."""

    @pytest.fixture(autouse=True)
    def _check_ultralytics(self):
        pytest.importorskip("ultralytics", reason="ultralytics not installed")

    def test_frame_by_frame_match(self):
        """Both trackers must produce the same tracks for the same input sequence."""
        our_frames, ultra_frames = _make_detections()
        ours = _run_ours(our_frames)
        ultra = _run_ultra(ultra_frames)

        assert len(ours) == len(ultra), "Different number of output frames"
        for i, (o, u) in enumerate(zip(ours, ultra)):
            o_sorted = _sort_by_track_id(o)
            u_sorted = _sort_by_track_id(u)
            assert o_sorted.shape == u_sorted.shape, (
                f"Frame {i}: shape mismatch ours={o_sorted.shape} ultra={u_sorted.shape}"
            )
            np.testing.assert_allclose(
                o_sorted, u_sorted, atol=1e-4,
                err_msg=f"Frame {i}: output mismatch",
            )

    def test_empty_detections(self):
        """Both trackers handle empty detections identically."""
        our_empty = np.empty((0, 6), dtype=np.float32)
        ultra_empty = _UltraResults(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        ours = _run_ours([our_empty] * 3)
        ultra = _run_ultra([ultra_empty] * 3)
        for i, (o, u) in enumerate(zip(ours, ultra)):
            assert o.shape == u.shape, f"Frame {i}: shape mismatch on empty input"

    def test_single_detection(self):
        """A single high-confidence detection should produce one track in both."""
        our_det = np.array([[200, 200, 50, 80, 0.95, 0]], dtype=np.float32)
        ultra_det = _UltraResults(
            np.array([[200, 200, 50, 80]], dtype=np.float32),
            np.array([0.95], dtype=np.float32),
            np.array([0], dtype=np.float32),
        )
        ours = _run_ours([our_det] * 5)
        ultra = _run_ultra([ultra_det] * 5)
        for i, (o, u) in enumerate(zip(ours, ultra)):
            o_sorted = _sort_by_track_id(o)
            u_sorted = _sort_by_track_id(u)
            assert o_sorted.shape == u_sorted.shape, (
                f"Frame {i}: shape mismatch ours={o_sorted.shape} ultra={u_sorted.shape}"
            )
            np.testing.assert_allclose(
                o_sorted, u_sorted, atol=1e-4,
                err_msg=f"Frame {i}: single-detection mismatch",
            )

    def test_disappearing_object(self):
        """An object that disappears mid-sequence should be handled identically."""
        our_frames = []
        ultra_frames = []
        for f in range(8):
            if f < 4:
                xywh = np.array([[100, 100, 40, 60], [300, 300, 50, 70]], dtype=np.float32)
                xywh[:, 0] += f * 5
                xywh[:, 1] += f * 3
                conf = np.array([0.9, 0.85], dtype=np.float32)
                cls = np.zeros(2, dtype=np.float32)
            else:
                xywh = np.array([[100 + f * 5, 100 + f * 3, 40, 60]], dtype=np.float32)
                conf = np.array([0.9], dtype=np.float32)
                cls = np.zeros(1, dtype=np.float32)
            our_frames.append(np.column_stack([xywh, conf, cls]))
            ultra_frames.append(_UltraResults(xywh, conf, cls))

        ours = _run_ours(our_frames)
        ultra = _run_ultra(ultra_frames)
        assert len(ours) == len(ultra)
        for i, (o, u) in enumerate(zip(ours, ultra)):
            o_sorted = _sort_by_track_id(o)
            u_sorted = _sort_by_track_id(u)
            assert o_sorted.shape == u_sorted.shape, (
                f"Frame {i}: shape mismatch ours={o_sorted.shape} ultra={u_sorted.shape}"
            )
            np.testing.assert_allclose(
                o_sorted, u_sorted, atol=1e-4,
                err_msg=f"Frame {i}: disappearing-object mismatch",
            )
