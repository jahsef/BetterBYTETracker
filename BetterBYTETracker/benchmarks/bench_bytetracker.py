"""Benchmark BetterBYTETracker vs ultralytics BYTETracker.

Usage:
    python -m BetterBYTETracker.benchmarks.bench_bytetracker
    python -m BetterBYTETracker.benchmarks.bench_bytetracker --sprints 128 --warmup 16 --frames 128 --multiplier 10
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


def _make_sequence(n_frames: int, multiplier: int, seed: int = 0):
    """Generate detection sequence with 4 detection types per multiplier.

    Detection types (each spawned `multiplier` times):
      - linear: left-to-right / right-to-left with occasional lost frames
      - erroneous: random short-lived detections popping in/out
      - circular: moves in a circle with occasional lost frames
      - random_walk: random movement, mean-reverting to stay centered

    Args:
        n_frames: number of frames to generate.
        multiplier: number of detections per type (total ~ 4 * multiplier per frame).
        seed: random seed.

    Returns:
        (our_frames, ultra_frames) lists of per-frame data.
    """
    rng = np.random.RandomState(seed)

    # --- Linear detections: half LTR, half RTL, 5% drop chance ---
    n_lin = multiplier
    n_ltr = n_lin // 2
    n_rtl = n_lin - n_ltr
    lin_start_x = np.concatenate([
        np.linspace(50, 150, n_ltr),
        np.linspace(600, 500, n_rtl),
    ]).astype(np.float32)
    lin_start_y = np.linspace(100, 300, n_lin).astype(np.float32)
    lin_vx = np.concatenate([np.full(n_ltr, 8.0), np.full(n_rtl, -8.0)]).astype(np.float32)
    lin_w = rng.uniform(30, 60, n_lin).astype(np.float32)
    lin_h = rng.uniform(50, 90, n_lin).astype(np.float32)
    lin_drop = 0.05

    # --- Circular detections: 5% drop chance ---
    n_circ = multiplier
    circ_cx = rng.uniform(200, 500, n_circ).astype(np.float32)
    circ_cy = rng.uniform(150, 350, n_circ).astype(np.float32)
    circ_r = rng.uniform(30, 80, n_circ).astype(np.float32)
    circ_phase = rng.uniform(0, 2 * np.pi, n_circ).astype(np.float32)
    circ_speed = rng.uniform(0.05, 0.15, n_circ).astype(np.float32)
    circ_w = rng.uniform(25, 50, n_circ).astype(np.float32)
    circ_h = rng.uniform(40, 80, n_circ).astype(np.float32)
    circ_drop = 0.05

    # --- Random walk detections: mean-reverting, 5% drop chance ---
    n_rw = multiplier
    rw_x = rng.uniform(150, 500, n_rw).astype(np.float32)
    rw_y = rng.uniform(100, 400, n_rw).astype(np.float32)
    rw_home_x = rw_x.copy()
    rw_home_y = rw_y.copy()
    rw_w = rng.uniform(30, 55, n_rw).astype(np.float32)
    rw_h = rng.uniform(50, 85, n_rw).astype(np.float32)
    rw_drop = 0.05
    rw_revert = 0.05  # mean reversion strength

    # --- Erroneous detections: random, 40% drop chance (short-lived) ---
    n_err = multiplier

    our_frames = []
    ultra_frames = []

    for f in range(n_frames):
        parts = []

        # Linear
        noise = rng.randn(n_lin, 2).astype(np.float32) * 1.5
        lin_alive = rng.random(n_lin) >= lin_drop
        if lin_alive.any():
            lin_xywh = np.column_stack([
                lin_start_x + f * lin_vx + noise[:, 0],
                lin_start_y + f * 2 + noise[:, 1],
                lin_w, lin_h,
            ])[lin_alive]
            lin_conf = np.clip(0.85 + rng.randn(int(lin_alive.sum())).astype(np.float32) * 0.04, 0.5, 1.0)
            parts.append(np.column_stack([lin_xywh, lin_conf, np.zeros(len(lin_conf), dtype=np.float32)]))

        # Circular
        circ_alive = rng.random(n_circ) >= circ_drop
        if circ_alive.any():
            angle = circ_phase + f * circ_speed
            cx = circ_cx + circ_r * np.cos(angle)
            cy = circ_cy + circ_r * np.sin(angle)
            noise_c = rng.randn(n_circ, 2).astype(np.float32) * 1.0
            circ_xywh = np.column_stack([
                cx + noise_c[:, 0], cy + noise_c[:, 1],
                circ_w, circ_h,
            ])[circ_alive]
            circ_conf = np.clip(0.82 + rng.randn(int(circ_alive.sum())).astype(np.float32) * 0.05, 0.5, 1.0)
            parts.append(np.column_stack([circ_xywh, circ_conf, np.ones(len(circ_conf), dtype=np.float32)]))

        # Random walk (mean-reverting)
        rw_alive = rng.random(n_rw) >= rw_drop
        rw_x += rng.randn(n_rw).astype(np.float32) * 5.0 + rw_revert * (rw_home_x - rw_x)
        rw_y += rng.randn(n_rw).astype(np.float32) * 5.0 + rw_revert * (rw_home_y - rw_y)
        if rw_alive.any():
            rw_xywh = np.column_stack([rw_x, rw_y, rw_w, rw_h])[rw_alive]
            rw_conf = np.clip(0.80 + rng.randn(int(rw_alive.sum())).astype(np.float32) * 0.06, 0.5, 1.0)
            parts.append(np.column_stack([rw_xywh, rw_conf, np.full(len(rw_conf), 2.0, dtype=np.float32)]))

        # Erroneous (random short-lived)
        err_alive = rng.random(n_err) >= 0.40
        if err_alive.any():
            n_alive_err = int(err_alive.sum())
            err_xywh = np.column_stack([
                rng.uniform(0, 640, n_alive_err).astype(np.float32),
                rng.uniform(0, 480, n_alive_err).astype(np.float32),
                rng.uniform(20, 60, n_alive_err).astype(np.float32),
                rng.uniform(30, 80, n_alive_err).astype(np.float32),
            ])
            err_conf = np.clip(0.55 + rng.randn(n_alive_err).astype(np.float32) * 0.15, 0.2, 1.0)
            parts.append(np.column_stack([err_xywh, err_conf, np.full(n_alive_err, 3.0, dtype=np.float32)]))

        if parts:
            frame = np.concatenate(parts, axis=0).astype(np.float32)
        else:
            frame = np.empty((0, 6), dtype=np.float32)

        our_frames.append(frame)
        ultra_frames.append(_UltraResults(frame[:, :4], frame[:, 4], frame[:, 5]))

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
    parser.add_argument("--multiplier", type=int, default=10, help="Detections per type (total ~ 4x this, default: 10)")
    args = parser.parse_args()

    n_sprints = args.sprints
    n_warmup = args.warmup
    n_frames = args.frames
    multiplier = args.multiplier

    our_frames, ultra_frames = _make_sequence(n_frames, multiplier)

    avg_dets = np.mean([len(f) for f in our_frames])

    def make_ours():
        return OursBYTETracker(TRACKER_ARGS, frame_rate=30)

    def make_ultra():
        return UltraBYTETracker(TRACKER_ARGS, frame_rate=30)

    n_samples = n_sprints * n_frames

    print(f"Config: {n_warmup} warmup + {n_sprints} sprints x {n_frames} frames/sprint, "
          f"~{avg_dets:.0f} detections/frame (multiplier={multiplier}), {n_samples} samples\n")

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
