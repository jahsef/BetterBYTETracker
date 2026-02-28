# BetterBYTETracker

Drop-in replacement for [ultralytics BYTETracker](https://github.com/ultralytics/ultralytics) that's ~2x faster. Same algorithm, same output, no extra dependencies.

The speedup comes from replacing per-track Python objects (`STrack`) with pure numpy allowing better vectorization.

## Benchmarks

10 objects/frame, 128 sprints x 128 frames

| Tracker | Mean (ms/frame) | CI95 |
|---|---|---|
| **ours** | **0.38** | 0.001 |
| ultralytics | 0.78 | 0.001 |

**2.07x faster.** Ryzen 7 5700x3d, Python 3.11, numpy 2.2.6

```
python -m BetterBYTETracker.benchmarks.bench_bytetracker --sprints 128 --warmup 16 --frames 128
```

## Installation
- Clone repo
- cd repo
- pip install .
- profit

## Limitations

- **Regular bounding boxes only** — no OBB (oriented bounding box) support
- **Batch numpy input** — expects numpy arrays, not Ultralytics Results object
- **API change** 

    ultralytics def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:

    Ours:       def update(self, detections: np.ndarray) -> np.ndarray
- **No ReID / appearance features** — purely IoU-based matching (same as upstream BYTETracker default)

## Tests

Parity test to verify same output

```
python -m pytest BetterBYTETracker/tests/test_parity.py -v
```
