import numpy as np
import scipy.optimize

from ...utils.metrics import bbox_ioa

try:
    import lap

    assert lap.__version__
except (ImportError, AssertionError, AttributeError):
    lap = None


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using scipy or lap.lapjv.

    Args:
        cost_matrix: (N, M) cost matrix.
        thresh: Threshold for valid assignments.
        use_lap: Use lap.lapjv if available.

    Returns:
        matched_indices: (K, 2) array of matches.
        unmatched_a: Unmatched row indices.
        unmatched_b: Unmatched column indices.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap and lap is not None:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(xyxy_a: np.ndarray, xyxy_b: np.ndarray) -> np.ndarray:
    """Compute IoU-based cost matrix between two sets of xyxy boxes.

    Args:
        xyxy_a: (N, 4) xyxy bounding boxes.
        xyxy_b: (M, 4) xyxy bounding boxes.

    Returns:
        (N, M) cost matrix = 1 - IoU.
    """
    if len(xyxy_a) == 0 or len(xyxy_b) == 0:
        return np.zeros((len(xyxy_a), len(xyxy_b)), dtype=np.float32)

    ious = bbox_ioa(
        np.ascontiguousarray(xyxy_a, dtype=np.float32),
        np.ascontiguousarray(xyxy_b, dtype=np.float32),
        iou=True,
    )
    return 1 - ious


def fuse_score(cost_matrix: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """Fuse cost matrix with detection scores.

    Args:
        cost_matrix: (N, M) cost matrix.
        det_scores: (M,) detection confidence scores.

    Returns:
        (N, M) fused cost matrix.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    scores_row = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * scores_row
    return 1 - fuse_sim
