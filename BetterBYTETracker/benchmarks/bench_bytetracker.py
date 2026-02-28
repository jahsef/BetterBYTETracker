"""Benchmark cuda-bytetracker vs ultralytics BYTETracker.

Usage:
    python -m cuda-bytetracker.benchmarks.bench_bytetracker
    python -m cuda-bytetracker.benchmarks.bench_bytetracker --sprints 128 --warmup 16 --frames 128
"""

from __future__ import annotations

import argparse
import importlib
import time
from types import SimpleNamespace

import numpy as np


# ---------------------------------------------------------------------------
# Fake results object
# ---------------------------------------------------------------------------

class FakeResults:
    """Numpy-backed results for both trackers."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return FakeResults(self.xywh[idx], self.conf[idx], self.cls[idx])


# ---------------------------------------------------------------------------
# Detection generators
# ---------------------------------------------------------------------------

def _make_sequence(n_frames: int, n_objects: int, seed: int = 0):
    """Objects where half move left-to-right and half move right-to-left.

    Returns list of FakeResults frames.
    """
    rng = np.random.RandomState(seed)
    n_ltr = n_objects // 2
    n_rtl = n_objects - n_ltr

    # LTR group starts on the left, RTL group starts on the right
    ltr_x = np.linspace(50, 150, n_ltr, dtype=np.float32)
    rtl_x = np.linspace(600, 500, n_rtl, dtype=np.float32)
    start_x = np.concatenate([ltr_x, rtl_x])
    start_y = np.linspace(100, 300, n_objects, dtype=np.float32)
    # Per-object x velocity: +8 for LTR, -8 for RTL
    vx = np.concatenate([np.full(n_ltr, 8.0), np.full(n_rtl, -8.0)]).astype(np.float32)
    w = rng.uniform(30, 60, n_objects).astype(np.float32)
    h = rng.uniform(50, 90, n_objects).astype(np.float32)

    frames = []
    for f in range(n_frames):
        noise = rng.randn(n_objects, 2).astype(np.float32) * 1.5
        xywh = np.column_stack([
            start_x + f * vx + noise[:, 0],
            start_y + f * 2 + noise[:, 1],
            w, h,
        ])
        conf = np.clip(0.85 + rng.randn(n_objects).astype(np.float32) * 0.04, 0.5, 1.0)
        cls = np.zeros(n_objects, dtype=np.float32)
        frames.append(FakeResults(xywh, conf, cls))
    return frames


# ---------------------------------------------------------------------------
# Tracker args
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
# Runner
# ---------------------------------------------------------------------------

def _time_tracker(make_tracker, frames, n_sprints: int, n_warmup: int) -> np.ndarray:
    """Return per-frame timings in seconds, shape (n_sprints * n_frames,).

    Runs n_warmup untimed sprints first to warm up caches and JIT paths.
    """
    n_frames = len(frames)

    # Warmup sprints (not timed)
    for _ in range(n_warmup):
        tracker = make_tracker()
        for det in frames:
            tracker.update(det)

    # Timed sprints
    frame_times = np.empty(n_sprints * n_frames, dtype=np.float64)
    idx = 0
    for _ in range(n_sprints):
        tracker = make_tracker()
        for det in frames:
            t0 = time.perf_counter()
            tracker.update(det)
            frame_times[idx] = time.perf_counter() - t0
            idx += 1
    return frame_times


# ---------------------------------------------------------------------------
# Stats / display
# ---------------------------------------------------------------------------

def _stats(times_s: np.ndarray) -> dict:
    """Compute mean, std, CI95 in milliseconds."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    frames = _make_sequence(n_frames, n_objects)

    # Sanity-check directions (first half LTR, second half RTL)
    n_ltr = n_objects // 2
    assert frames[-1].xywh[0, 0] > frames[0].xywh[0, 0], "LTR objects: x should increase"
    assert frames[-1].xywh[n_ltr, 0] < frames[0].xywh[n_ltr, 0], "RTL objects: x should decrease"

    # -- Our tracker --
    ours_mod = importlib.import_module("cuda-bytetracker.trackers.byte_tracker")
    OursBYTETracker = ours_mod.BYTETracker

    def make_ours():
        return OursBYTETracker(TRACKER_ARGS, frame_rate=30)

    # -- Ultralytics tracker --
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker as UltraBYTETracker

        def make_ultra():
            return UltraBYTETracker(TRACKER_ARGS, frame_rate=30)
        has_ultra = True
    except ImportError:
        has_ultra = False

    n_samples = n_sprints * n_frames

    print(f"Config: {n_warmup} warmup + {n_sprints} sprints x {n_frames} frames/sprint, "
          f"{n_objects} objects/frame ({n_ltr} LTR + {n_objects - n_ltr} RTL), "
          f"{n_samples} per-frame samples\n")

    rows = []

    times = _time_tracker(make_ours, frames, n_sprints, n_warmup)
    rows.append({"label": "ours", **_stats(times)})

    if has_ultra:
        times = _time_tracker(make_ultra, frames, n_sprints, n_warmup)
        rows.append({"label": "ultralytics", **_stats(times)})
    else:
        print("(ultralytics not installed, skipping comparison)\n")

    _print_table(rows)

    # Speedup summary
    if has_ultra and len(rows) == 2:
        ours_row, ultra_row = rows[0], rows[1]
        if ultra_row["mean_ms"] > 0:
            ratio = ultra_row["mean_ms"] / ours_row["mean_ms"]
            print(f"\nours is {ratio:.2f}x vs ultralytics")


if __name__ == "__main__":
    main()
