import numpy as np
import scipy

from ...utils.metrics import bbox_ioa

try:
    import lap

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    lap = None


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (list[list[int]] | np.ndarray): Matched indices of shape (K, 2), where K is the number of
            matches.
        unmatched_a (tuple | list | np.ndarray): Unmatched indices from the first set.
        unmatched_b (tuple | list | np.ndarray): Unmatched indices from the second set.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap and lap is not None:
        # Use lap.lapjv
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_ioa(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
            iou=True,
        )
    return 1 - ious  # cost matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single cost matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused cost matrix with shape (N, M).
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = det_scores[None].repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
