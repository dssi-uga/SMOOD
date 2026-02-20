import numpy as np
import joblib

EPS = 1e-12


def _w1_quantile(u_sorted, v_sorted, q_size=128):
    n, m = u_sorted.shape[0], v_sorted.shape[0]
    qs = int(max(q_size, min(512, max(n, m))))
    p = (np.arange(qs, dtype=np.float64) + 0.5) / qs
    pu = (np.arange(n, dtype=np.float64) + 0.5) / n
    pv = (np.arange(m, dtype=np.float64) + 0.5) / m
    uq = np.interp(p, pu, u_sorted)
    vq = np.interp(p, pv, v_sorted)
    return float(np.mean(np.abs(uq - vq)))


def swd_with_uncertainty(ref_proj_sorted, live_proj_sorted, q_size=128):
    """Calculates Mean SWD and Standard Error across projections."""
    K = ref_proj_sorted.shape[1]
    slice_dists = []
    for k in range(K):
        d = _w1_quantile(ref_proj_sorted[:, k], live_proj_sorted[:, k], q_size=q_size)
        slice_dists.append(d)

    slice_dists = np.array(slice_dists)
    mean_val = np.mean(slice_dists)
    # Standard Error of the Mean (SEM) represents estimation uncertainty
    uncertainty = np.std(slice_dists) / np.sqrt(K)
    return float(mean_val), float(uncertainty)


class FlorenceVisionOOD:
    def __init__(self, npz_path, pca_path="OOD/VisualOOD/pca_model.joblib", device="cpu"):
        d = np.load(npz_path, allow_pickle=True)

        self.ref_proj_sorted = d["ref_proj_sorted"].astype(np.float64)
        self.thetas = d["thetas"].astype(np.float64)
        self.threshold = float(d["threshold"])
        self.window = int(d["window"]) if "window" in d else 10
        self.n_projections = int(d["n_projections"]) if "n_projections" in d else self.thetas.shape[0]

        # New Standardization Parameters
        self.bias_offset = float(d["bias_offset"])
        self.mu_id = float(d["mu_id"])
        self.sigma_id = float(d["sigma_id"])

        # PCA
        self.use_pca = bool(int(d["use_pca"])) if "use_pca" in d else ("pca_path" in d)
        self.pca = None
        if self.use_pca:
            if "pca_path" in d:
                pca_path = str(d["pca_path"])
            if not pca_path:
                raise ValueError("use_pca=1 but pca_path is empty/missing.")
            self.pca = joblib.load(pca_path)

        self.device = device
        self._buf = []

    def reset(self):
        self._buf = []

    def update_and_score(self, latent):
        if hasattr(latent, "detach"):
            latent = latent.detach().cpu().numpy()

        x = np.asarray(latent, dtype=np.float64).reshape(1, -1)
        if self.use_pca:
            x_low = self.pca.transform(x).reshape(-1)  # (d_low,)
        else:
            x_low = (x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)).reshape(-1)

        self._buf.append(x_low)
        if len(self._buf) > self.window:
            self._buf.pop(0)

        if len(self._buf) < self.window:
            return None, None

        live = np.asarray(self._buf, dtype=np.float64)          # (W, d_low)
        live_proj = live @ self.thetas.T                        # (W, K)
        live_proj_sorted = np.sort(live_proj, axis=0)

        raw_score, raw_sigma = swd_with_uncertainty(self.ref_proj_sorted, live_proj_sorted)

        # Apply Z-Score Standardization
        centered = raw_score - self.bias_offset
        z_score = (centered - self.mu_id) / (self.sigma_id + EPS)
        z_sigma = raw_sigma / (self.sigma_id + EPS)

        return float(z_score), float(z_sigma)

    def is_ood_from_score(self, z_score, z_sigma):
        if z_score is None:
            return False
        return (z_score - z_sigma) > self.threshold

    def __repr__(self):
        return f"FlorenceVisionOOD(SWD, window={self.window}, K={self.n_projections}, thr={self.threshold:.6f}, pca={self.use_pca})"

