import os
import sys

# Allow imports from SMOOD_GitHub/* when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.decomposition import PCA
import joblib

# ========================= PATHS =========================
LATENT_PATH = "OOD/VisualOOD/vision_latents_L2_normal.npy"
OUT_PATH = "OOD/VisualOOD/vision_ood_model.npz"
PCA_PATH = "OOD/VisualOOD/pca_model.joblib"

# ========================= CONFIG =========================
USE_PCA = False
N_COMPONENTS = 96
WINDOW = 30
N_PROJECTIONS = 256
MAX_REF = 10000
STRIDE = 5
SEED = 123
PERCENTILE_THR = 99.5

EPS = 1e-12


def _make_thetas(d, k, seed=123):
    rng = np.random.default_rng(seed)
    flat_matrix = rng.normal(size=(d, k)).astype(np.float64)
    q, _ = np.linalg.qr(flat_matrix)
    return q.T


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
    K = ref_proj_sorted.shape[1]
    slice_dists = []
    for k in range(K):
        d = _w1_quantile(ref_proj_sorted[:, k], live_proj_sorted[:, k], q_size=q_size)
        slice_dists.append(d)

    slice_dists = np.array(slice_dists)
    return float(np.mean(slice_dists)), float(np.std(slice_dists) / np.sqrt(K))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[INFO] Loading latents:", LATENT_PATH)
    X = np.load(LATENT_PATH)
    N, D = X.shape

    pca_path_saved = ""
    if USE_PCA:
        pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
        X_low = pca.fit_transform(X).astype(np.float64)
        joblib.dump(pca, PCA_PATH)
        pca_path_saved = PCA_PATH
        d_low = N_COMPONENTS
    else:
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        X_low = X_norm.astype(np.float64)
        d_low = D

    X_ref = X_low[rng.choice(N, size=min(N, MAX_REF), replace=False)]
    thetas = _make_thetas(d_low, N_PROJECTIONS, seed=SEED)
    ref_proj_sorted = np.sort(X_ref @ thetas.T, axis=0).astype(np.float64)

    print("[INFO] Calculating Empirical Bias Offset...")
    bias_samples = []
    for _ in range(500):
        idx1 = rng.choice(N, size=WINDOW, replace=False)
        idx2 = rng.choice(N, size=WINDOW, replace=False)
        p1 = np.sort(X_low[idx1] @ thetas.T, axis=0)
        p2 = np.sort(X_low[idx2] @ thetas.T, axis=0)
        s_bias, _ = swd_with_uncertainty(p1, p2)
        bias_samples.append(s_bias)
    bias_offset = float(np.mean(bias_samples))

    print("[INFO] Calibrating Standardized Scores...")
    raw_scores = []
    for i in range(0, len(X_low) - WINDOW, STRIDE):
        live_proj_sorted = np.sort(X_low[i:i + WINDOW] @ thetas.T, axis=0)
        s, _ = swd_with_uncertainty(ref_proj_sorted, live_proj_sorted)
        raw_scores.append(s)

    raw_scores = np.asarray(raw_scores)
    centered = raw_scores - bias_offset

    mu_id = float(np.mean(centered))
    sigma_id = float(np.std(centered))

    z_scores = (centered - mu_id) / (sigma_id + EPS)
    thr_z = float(np.percentile(z_scores, PERCENTILE_THR))

    print("-" * 30)
    print(f"[RESULT] Bias Offset: {bias_offset:.6f}")
    print(f"[RESULT] Nominal Mean:   {mu_id:.6f}")
    print(f"[RESULT] Nominal Std:    {sigma_id:.6f}")
    print(f"[RESULT] Z-Threshold: {thr_z:.2f} sigma")
    print("-" * 30)

    np.savez(
        OUT_PATH,
        ref_proj_sorted=ref_proj_sorted.astype(np.float32),
        thetas=thetas.astype(np.float32),
        threshold=np.float32(thr_z),      # Z-threshold
        bias_offset=np.float32(bias_offset),
        mu_id=np.float32(mu_id),          # For live centering
        sigma_id=np.float32(sigma_id),    # For live scaling
        window=np.int32(WINDOW),
        n_projections=np.int32(N_PROJECTIONS),
        use_pca=np.int32(1 if USE_PCA else 0),
        pca_path=np.array(pca_path_saved)
    )
    print("[INFO] Saved Standardized Model to:", OUT_PATH)


if __name__ == "__main__":
    main()

