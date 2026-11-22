# Copyright (c) the PUF authors (Yi Yang, Myrna Castillo, Bodo Rosenhahn,
# Michael Ying Yang).  Adapted from https://github.com/yyyyangyi/PUF
# (Merging/puf_utils.py, Merging/utils.py), Apache License 2.0.
# Modifications for WSGG: restructured into free functions; the covariance
# determinant ratio is computed in log space and guarded against singular
# covariances (monocular Pi3 point clusters can be degenerate).
"""Vendored PUF / FROSS association math.

* :func:`hellinger`            Bhattacharyya/Hellinger distance, one Gaussian vs many
                               (PUF_SG._batched_hellinger_distance, identical formula).
* :func:`semantic_likelihood`  exp(-JSD(q_obs, q_cand) / sigma) on Dirichlet means
                               (PUF_SG._compute_likelihood_row, semantic factor).
* :func:`association_likelihood` spatial (1 - H^2) x semantic (PUF Eq. for L).
* :func:`jpda_marginals`       beta_d = L / (lambda_birth + sum L), beta_birth
                               (PUF_SG._jpda_marginals_row).
* :func:`merge_gaussians`      point-count-weighted moment matching
                               (GaussianSG._merge_gaussians / PUF_SG.fuse hard geometry update).
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy


def hellinger(mean1: np.ndarray, cov1: np.ndarray, means2: np.ndarray, covs2: np.ndarray) -> np.ndarray:
    """Hellinger distance between N(mean1, cov1) and each N(means2[k], covs2[k]) -> (K,)."""
    m1 = mean1[None, :, None]
    m2 = means2[..., None]
    c1 = cov1[None]
    diff = m1 - m2
    cov_mean = (c1 + covs2) / 2.0
    cov_mean_inv = np.linalg.inv(cov_mean)
    _, logdet_mean = np.linalg.slogdet(cov_mean)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(covs2)
    b_d = (0.125 * diff.transpose(0, 2, 1) @ cov_mean_inv @ diff).reshape(-1) \
        + 0.5 * (logdet_mean - 0.5 * (logdet1 + logdet2))
    return np.sqrt(np.clip(1.0 - np.exp(-b_d), 0.0, 1.0))


def semantic_likelihood(obs_alpha: np.ndarray, cand_alphas: np.ndarray, sigma_jsd: float) -> np.ndarray:
    obs_s = obs_alpha.sum()
    obs_q = obs_alpha / (obs_s if obs_s > 0 else 1.0)
    cand_s = cand_alphas.sum(axis=1)
    cand_q = cand_alphas / np.where(cand_s[:, None] > 0, cand_s[:, None], 1.0)
    mix = (obs_q[None] + cand_q) / 2.0
    safe = np.where(mix > 0, mix, 1.0)
    jsd = 0.5 * (xlogy(obs_q[None], obs_q[None] / safe).sum(axis=1)
                 + xlogy(cand_q, cand_q / safe).sum(axis=1))
    return np.exp(-jsd / sigma_jsd)


def association_likelihood(obs_mean, obs_cov, obs_alpha, cand_means, cand_covs, cand_alphas,
                           sigma_jsd: float, has_geom: np.ndarray = None) -> np.ndarray:
    """L = (1 - H^2) * exp(-JSD / sigma).  ``obs_mean is None`` (no Pi3 geometry
    for this observation) or ``has_geom[k] == False`` drop the spatial factor
    (set to 1) -- a WSGG deviation, PUF itself discards depth-less observations."""
    l_sem = semantic_likelihood(obs_alpha, cand_alphas, sigma_jsd)
    if obs_mean is None:
        return l_sem
    l_sp = np.ones(len(cand_alphas))
    idx = np.arange(len(cand_alphas)) if has_geom is None else np.nonzero(has_geom)[0]
    if len(idx):
        h = hellinger(obs_mean, obs_cov, cand_means[idx], cand_covs[idx])
        l_sp[idx] = 1.0 - h ** 2
    return l_sp * l_sem


def jpda_marginals(L: np.ndarray, lambda_birth: float):
    z = lambda_birth + L.sum()
    return L / z, lambda_birth / z


def merge_gaussians(mean1, cov1, n1, mean2, cov2, n2):
    """Moment-matched merge of two Gaussians weighted by point counts."""
    if n1 == 0 and n2 == 0:
        n1 = n2 = 1
    tot = float(n1 + n2)
    diff = mean1 - mean2
    mean = (n1 * mean1 + n2 * mean2) / tot
    cov = (n1 * cov1 + n2 * cov2) / tot + n1 * n2 * np.outer(diff, diff) / tot ** 2
    return mean, cov
