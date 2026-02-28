import numpy as np


class KalmanFilterXYAH:
    """Kalman filter for tracking bounding boxes in image space.

    8-dimensional state space (x, y, a, h, vx, vy, va, vh) with constant velocity model.
    Bounding box location (x, y, a, h) is the direct observation (linear observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # ------------------------------------------------------------------
    # Single-track methods (kept for reference / debugging)
    # ------------------------------------------------------------------

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = np.stack([
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            np.full((), 1e-2),
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            np.full((), 1e-5),
            10 * self._std_weight_velocity * h,
        ])
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std_pos = np.stack([
            self._std_weight_position * h,
            self._std_weight_position * h,
            np.full((), 1e-2),
            self._std_weight_position * h,
        ])
        std_vel = np.stack([
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            np.full((), 1e-5),
            self._std_weight_velocity * h,
        ])
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.dot(np.dot(self._motion_mat, covariance), self._motion_mat.T) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std = np.stack([
            self._std_weight_position * h,
            self._std_weight_position * h,
            np.full((), 1e-1),
            self._std_weight_position * h,
        ])
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.dot(np.dot(self._update_mat, covariance), self._update_mat.T)
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)

        L = np.linalg.cholesky(projected_cov)
        rhs = np.dot(covariance, self._update_mat.T).T
        kalman_gain = np.linalg.solve(L.T, np.linalg.solve(L, rhs)).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.dot(np.dot(kalman_gain, projected_cov), kalman_gain.T)
        return new_mean, new_covariance

    # ------------------------------------------------------------------
    # Batch methods — operate on (N, 8), (N, 8, 8) arrays
    # ------------------------------------------------------------------

    def batch_initiate(self, measurements: np.ndarray):
        """Create tracks from unassociated measurements.

        Args:
            measurements: (M, 4) array in (x, y, a, h) format.

        Returns:
            means: (M, 8) mean vectors.
            covariances: (M, 8, 8) covariance matrices.
        """
        M = len(measurements)
        if M == 0:
            return np.empty((0, 8), dtype=np.float32), np.empty((0, 8, 8), dtype=np.float32)

        mean_vel = np.zeros((M, 4), dtype=np.float32)
        means = np.concatenate([measurements, mean_vel], axis=1)

        h = measurements[:, 3]  # (M,)
        swp = self._std_weight_position
        swv = self._std_weight_velocity

        # Build (M, 8) std array
        std = np.stack([
            2 * swp * h,
            2 * swp * h,
            np.full(M, 1e-2, dtype=np.float32),
            2 * swp * h,
            10 * swv * h,
            10 * swv * h,
            np.full(M, 1e-5, dtype=np.float32),
            10 * swv * h,
        ], axis=1)  # (M, 8)

        sqr = np.square(std)  # (M, 8)
        covariances = np.zeros((M, 8, 8), dtype=np.float32)
        idx = np.arange(8)
        covariances[:, idx, idx] = sqr

        return means, covariances

    def batch_predict(self, means: np.ndarray, covariances: np.ndarray):
        """Batch Kalman predict. Equivalent to multi_predict but returns results.

        Args:
            means: (N, 8) mean vectors.
            covariances: (N, 8, 8) covariance matrices.

        Returns:
            means: (N, 8) predicted means.
            covariances: (N, 8, 8) predicted covariances.
        """
        N = len(means)
        if N == 0:
            return means, covariances

        h = means[:, 3]
        swp = self._std_weight_position
        swv = self._std_weight_velocity

        std = np.stack([
            swp * h,
            swp * h,
            np.full(N, 1e-2, dtype=np.float32),
            swp * h,
            swv * h,
            swv * h,
            np.full(N, 1e-5, dtype=np.float32),
            swv * h,
        ], axis=1)  # (N, 8)

        sqr = np.square(std)
        motion_cov = np.zeros((N, 8, 8), dtype=np.float32)
        idx = np.arange(8)
        motion_cov[:, idx, idx] = sqr

        # mean = mean @ motion_mat.T  — broadcasted
        means = np.dot(means, self._motion_mat.T)
        # cov = F @ cov @ F.T + Q  — batch matmul via einsum
        # F @ cov: (8,8) @ (N,8,8) -> use broadcasting
        left = np.einsum('ij,njk->nik', self._motion_mat, covariances)
        covariances = np.einsum('nij,kj->nik', left, self._motion_mat) + motion_cov

        return means, covariances

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Vectorized predict for multiple tracks (legacy interface)."""
        return self.batch_predict(mean, covariance)

    def batch_project(self, means: np.ndarray, covariances: np.ndarray):
        """Project state distribution to measurement space (batch).

        Args:
            means: (N, 8) state means.
            covariances: (N, 8, 8) state covariances.

        Returns:
            proj_means: (N, 4) projected means.
            proj_covs: (N, 4, 4) projected covariances (with innovation noise).
        """
        N = len(means)
        h = means[:, 3]
        swp = self._std_weight_position

        std = np.stack([
            swp * h,
            swp * h,
            np.full(N, 1e-1, dtype=np.float32),
            swp * h,
        ], axis=1)  # (N, 4)

        innovation_var = np.square(std)  # (N, 4)
        innovation_cov = np.zeros((N, 4, 4), dtype=np.float32)
        idx4 = np.arange(4)
        innovation_cov[:, idx4, idx4] = innovation_var

        # H @ mean: (4,8) @ (N,8).T -> (N,4)
        proj_means = np.dot(means, self._update_mat.T)
        # H @ cov @ H.T: einsum
        left = np.einsum('ij,njk->nik', self._update_mat, covariances)
        proj_covs = np.einsum('nij,kj->nik', left, self._update_mat) + innovation_cov

        return proj_means, proj_covs

    def batch_update(self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray):
        """Batch Kalman update.

        Args:
            means: (K, 8) predicted state means.
            covariances: (K, 8, 8) predicted state covariances.
            measurements: (K, 4) measurement vectors.

        Returns:
            new_means: (K, 8) corrected means.
            new_covariances: (K, 8, 8) corrected covariances.
        """
        K = len(means)
        if K == 0:
            return means, covariances

        proj_means, proj_covs = self.batch_project(means, covariances)

        # Kalman gain via solve: K = cov @ H.T @ inv(S)
        # cov @ H.T: (K,8,8) @ (8,4) -> (K,8,4)
        cov_Ht = np.einsum('nij,kj->nik', covariances, self._update_mat)  # (K, 8, 4)

        # Solve S @ K_gain.T = (cov @ H.T).T  for each slice
        # Use: kalman_gain = (S^{-1} @ cov_Ht^T)^T = solve(S, cov_Ht.transpose(0,2,1)).transpose(0,2,1)
        rhs = cov_Ht.transpose(0, 2, 1)  # (K, 4, 8)
        kalman_gain = np.linalg.solve(proj_covs, rhs).transpose(0, 2, 1)  # (K, 8, 4)

        innovation = measurements - proj_means  # (K, 4)

        # new_mean = mean + innovation @ kalman_gain.T  — but kalman_gain is (K,8,4)
        # innovation (K,4) @ kalman_gain.T would be (K,4) @ (K,4,8) which needs einsum
        new_means = means + np.einsum('ni,nji->nj', innovation, kalman_gain)

        # new_cov = cov - K @ S @ K.T
        # K @ S: (K,8,4) @ (K,4,4) -> einsum
        KS = np.einsum('nij,njk->nik', kalman_gain, proj_covs)  # (K, 8, 4)
        # KS @ K.T: (K,8,4) @ (K,8,4).T -> (K,8,8)
        new_covariances = covariances - np.einsum('nij,nkj->nik', KS, kalman_gain)

        return new_means, new_covariances

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError("Invalid distance metric")
