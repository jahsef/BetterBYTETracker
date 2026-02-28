import numpy as np


def xywh2ltwh(x):
    """Convert (x_center, y_center, w, h) to (x_topleft, y_topleft, w, h). Works on any shape (..., 4)."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def batch_tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Convert (N, 4) tlwh to (N, 4) xyah (center-x, center-y, aspect, height)."""
    ret = tlwh.copy()
    ret[:, :2] += ret[:, 2:] / 2  # center
    ret[:, 2] /= ret[:, 3]        # aspect = w / h
    return ret


def batch_means_to_xyxy(means: np.ndarray) -> np.ndarray:
    """Extract xyxy bounding boxes from Kalman state means (N, 8).

    State is (x, y, a, h, vx, vy, va, vh) where x,y = center, a = aspect, h = height.
    """
    cx = means[:, 0]
    cy = means[:, 1]
    a = means[:, 2]
    h = means[:, 3]
    w = a * h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def batch_means_to_tlwh(means: np.ndarray) -> np.ndarray:
    """Extract tlwh bounding boxes from Kalman state means (N, 8)."""
    cx = means[:, 0]
    cy = means[:, 1]
    a = means[:, 2]
    h = means[:, 3]
    w = a * h
    x1 = cx - w / 2
    y1 = cy - h / 2
    return np.stack([x1, y1, w, h], axis=1)
