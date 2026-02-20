import os
import sys

# Allow imports from SMOOD_GitHub/* when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.decomposition import PCA
import joblib

# ========================= PATHS =========================
LATENT_PATH = "OOD/ProprioOOD/ppo_latents_L2_normal.npy"
OUT_PATH = "OOD/ProprioOOD/ppo_swd_ood_model.npz"
PCA_PATH = "OOD/ProprioOOD/pca_model_ppo.joblib"

# ========================= CONFIG =========================
USE_PCA = False
N_COMPONENTS = 64      # Lower dim for PPO latents
WINDOW = 30
N_PROJECTIONS = 128    # K=128 is optimal for 128-D PPO space
MAX_REF = 10000
STRIDE = 5
SEED = 123
PERCENTILE_THR = 99.5

EPS = 1e-12


def _make_thetas(d, k, seed=123):
    """Generates semi-orthogonal projections via QR for full geometric coverage."""
    rng = np.random.default_rng(seed)
    flat_matrix = rng.normal(size=(d, k)).astype(np.float64)
    q, _ = np.linalg.qr(flat_matrix)
    return q.T


def _w1_quantile(u_sorted, v_sorted, q_size=128):
    """Mid-point quantile interpolation for W1 distance calculation."""
    n, m = u_sorted.shape[0], v_sorted.shape[0]
    qs = int(max(q_size, min(512, max(n, m))))
    p = (np.arange(qs, dtype=np.float64) + 0.5) / qs
    pu = (np.arange(n, dtype=np.float64) + 0.5) / n
    pv = (np.arange(m, dtype=np.float64) + 0.5) / m
    uq = np.interp(p, pu, u_sorted)
    vq = np.interp(p, pv, v_sorted)
    return float(np.mean(np.abs(uq - vq)))


def swd_with_uncertainty(ref_proj_sorted, live_proj_sorted, q_size=128):
    """Calculates Mean SWD and Standard Error of the Mean (SEM) across slices."""
    K = ref_proj_sorted.shape[1]
    slice_dists = []
    for k in range(K):
        d = _w1_quantile(ref_proj_sorted[:, k], live_proj_sorted[:, k], q_size=q_size)
        slice_dists.append(d)

    slice_dists = np.array(slice_dists)
    mean_score = np.mean(slice_dists)
    # Standard Error (SEM) allows for uncertainty quantification in the estimate
    uncertainty = np.std(slice_dists) / np.sqrt(K)
    return float(mean_score), float(uncertainty)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("[INFO] Loading PPO Proprioceptive latents:", LATENT_PATH)
    X = np.load(LATENT_PATH)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (N,D). Got shape {X.shape}")
    N, D = X.shape
    print("[INFO] Latents shape:", X.shape)

    # 1) Dimensionality Reduction / Normalization
    pca_path_saved = ""
    if USE_PCA:
        print(f"[INFO] Fitting PCA {D} → {N_COMPONENTS} ...")
        pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
        X_low = pca.fit_transform(X).astype(np.float64)
        joblib.dump(pca, PCA_PATH)
        pca_path_saved = PCA_PATH
        d_low = N_COMPONENTS
    else:
        # PPO latents benefit significantly from L2 normalization for SWD
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        X_low = X_norm.astype(np.float64)
        d_low = D
        print("[INFO] PCA disabled, using L2-normalized latents.")

    # 2) Reference set building
    X_ref = X_low[rng.choice(N, size=min(N, MAX_REF), replace=False)]
    thetas = _make_thetas(d_low, N_PROJECTIONS, seed=SEED)
    ref_proj_sorted = np.sort(X_ref @ thetas.T, axis=0).astype(np.float64)

    # 3) Calculate Empirical Bias Offset (Null Distribution)
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

    # 4) Calibrate Z-Scores on ID windows
    print("[INFO] Calibrating Standardized Scores...")
    raw_scores = []
    for i in range(0, len(X_low) - WINDOW, STRIDE):
        live_proj_sorted = np.sort(X_low[i:i + WINDOW] @ thetas.T, axis=0)
        s, _ = swd_with_uncertainty(ref_proj_sorted, live_proj_sorted)
        raw_scores.append(s)

    raw_scores = np.asarray(raw_scores)
    centered = raw_scores - bias_offset

    # Calculate ID mean and std for the Z-score transformation
    mu_id = float(np.mean(centered))
    sigma_id = float(np.std(centered))

    # Transform to Z-space (Standard Deviations)
    z_scores = (centered - mu_id) / (sigma_id + EPS)
    thr_z = float(np.percentile(z_scores, PERCENTILE_THR))

    print("-" * 30)
    print(f"[RESULT] PPO Bias Offset: {bias_offset:.6f}")
    print(f"[RESULT] Nominal Mean:   {mu_id:.6f}")
    print(f"[RESULT] Nominal Std:    {sigma_id:.6f}")
    print(f"[RESULT] Z-Threshold:    {thr_z:.2f} sigma")
    print("-" * 30)

    # 5) Save Standardized SWD model
    np.savez(
        OUT_PATH,
        ref_proj_sorted=ref_proj_sorted.astype(np.float32),
        thetas=thetas.astype(np.float32),
        threshold=np.float32(thr_z),
        bias_offset=np.float32(bias_offset),
        mu_id=np.float32(mu_id),          # Required for live centering
        sigma_id=np.float32(sigma_id),    # Required for live scaling
        window=np.int32(WINDOW),
        n_projections=np.int32(N_PROJECTIONS),
        use_pca=np.int32(1 if USE_PCA else 0),
        pca_path=np.array(pca_path_saved)
    )
    print("[INFO] Saved Standardized PPO SWD OOD model to:", OUT_PATH)


if __name__ == "__main__":
    main()

