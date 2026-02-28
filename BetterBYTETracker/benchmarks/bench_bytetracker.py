"""Benchmark BetterBYTETracker vs ultralytics BYTETracker.

Usage:
    python -m BetterBYTETracker.benchmarks.bench_bytetracker
    python -m BetterBYTETracker.benchmarks.bench_bytetracker --sprints 128 --warmup 16 --frames 128
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np

from ..trackers.byte_tracker import BYTETracker as OursBYTETracker
from ultralytics.trackers.byte_tracker import BYTETracker as UltraBYTETracker


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


def _make_sequence(n_frames: int, n_objects: int, seed: int = 0):
    """Objects where half move left-to-right and half move right-to-left."""
    rng = np.random.RandomState(seed)
    n_ltr = n_objects // 2
    n_rtl = n_objects - n_ltr

    ltr_x = np.linspace(50, 150, n_ltr, dtype=np.float32)
    rtl_x = np.linspace(600, 500, n_rtl, dtype=np.float32)
    start_x = np.concatenate([ltr_x, rtl_x])
    start_y = np.linspace(100, 300, n_objects, dtype=np.float32)
    vx = np.concatenate([np.full(n_ltr, 8.0), np.full(n_rtl, -8.0)]).astype(np.float32)
    w = rng.uniform(30, 60, n_objects).astype(np.float32)
    h = rng.uniform(50, 90, n_objects).astype(np.float32)

    our_frames = []
    ultra_frames = []
    for f in range(n_frames):
        noise = rng.randn(n_objects, 2).astype(np.float32) * 1.5
        xywh = np.column_stack([
            start_x + f * vx + noise[:, 0],
            start_y + f * 2 + noise[:, 1],
            w, h,
        ])
        conf = np.clip(0.85 + rng.randn(n_objects).astype(np.float32) * 0.04, 0.5, 1.0)
        cls = np.zeros(n_objects, dtype=np.float32)
        our_frames.append(np.column_stack([xywh, conf, cls]))
        ultra_frames.append(_UltraResults(xywh, conf, cls))
    return our_frames, ultra_frames


TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


def _time_tracker(make_tracker, frames, n_sprints: int, n_warmup: int) -> np.ndarray:
    """Return per-frame timings in seconds, shape (n_sprints * n_frames,)."""
    n_frames = len(frames)

    for _ in range(n_warmup):
        tracker = make_tracker()
        for f in range(n_frames):
            tracker.update(frames[f])

    frame_times = np.empty(n_sprints * n_frames, dtype=np.float64)
    idx = 0
    for _ in range(n_sprints):
        tracker = make_tracker()
        for f in range(n_frames):
            t0 = time.perf_counter()
            tracker.update(frames[f])
            frame_times[idx] = time.perf_counter() - t0
            idx += 1
    return frame_times


def _stats(times_s: np.ndarray) -> dict:
    ms = times_s * 1000
    n = len(ms)
    mean = float(np.mean(ms))
    std = float(np.std(ms, ddof=1)) if n > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"mean_ms": mean, "std_ms": std, "ci95_ms": ci95}


def _print_table(rows: list[dict]):
    header = f"{'Label':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'CI95 (ms)':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<35} {r['mean_ms']:>10.3f} {r['std_ms']:>10.3f} {r['ci95_ms']:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BYTETracker implementations")
    parser.add_argument("--sprints", type=int, default=5, help="Number of timed sprints (default: 5)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup sprints before timing (default: 3)")
    parser.add_argument("--frames", type=int, default=30, help="Frames per sprint (default: 30)")
    parser.add_argument("--objects", type=int, default=10, help="Objects per frame (default: 10)")
    args = parser.parse_args()

    n_sprints = args.sprints
    n_warmup = args.warmup
    n_frames = args.frames
    n_objects = args.objects

    our_frames, ultra_frames = _make_sequence(n_frames, n_objects)

    n_ltr = n_objects // 2
    assert our_frames[-1][0, 0] > our_frames[0][0, 0], "LTR objects: x should increase"
    assert our_frames[-1][n_ltr, 0] < our_frames[0][n_ltr, 0], "RTL objects: x should decrease"

    def make_ours():
        return OursBYTETracker(TRACKER_ARGS, frame_rate=30)

    def make_ultra():
        return UltraBYTETracker(TRACKER_ARGS, frame_rate=30)

    n_samples = n_sprints * n_frames

    print(f"Config: {n_warmup} warmup + {n_sprints} sprints x {n_frames} frames/sprint, "
          f"{n_objects} objects/frame, {n_samples} samples\n")

    rows = []

    times = _time_tracker(make_ours, our_frames, n_sprints, n_warmup)
    rows.append({"label": "ours", **_stats(times)})

    times = _time_tracker(make_ultra, ultra_frames, n_sprints, n_warmup)
    rows.append({"label": "ultralytics", **_stats(times)})

    _print_table(rows)

    ours_row, ultra_row = rows[0], rows[1]
    if ultra_row["mean_ms"] > 0:
        ratio = ultra_row["mean_ms"] / ours_row["mean_ms"]
        print(f"\nours is {ratio:.2f}x vs ultralytics")


if __name__ == "__main__":
    main()
