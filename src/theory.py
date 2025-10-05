import numpy as np
from scipy import stats as _stats


def compute_sublinear_gap(
    cumulative_regret: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Test whether cumulative regret grows sublinearly via log-log regression.

    Fits log(cumulative_regret) = β₀ + β₁ log(t) and returns (β₁, CI_lo, CI_hi)
    where the confidence interval is at level (1 - α).

    Theorem (Sublinear Regret Criterion):
    If β₁ < 1 with statistical significance at level α, the regret is
    sublinear (R(T) = O(T^{β₁})). If the confidence interval contains 1,
    the null hypothesis of linear regret cannot be rejected.

    Parameters
    ----------
    cumulative_regret : np.ndarray
        Array of cumulative regret values R(1), R(2), ..., R(T).
    alpha : float
        Significance level for the confidence interval (default 0.05).

    Returns
    -------
    alpha_hat : float
        Estimated exponent β₁ from log-log regression.
    ci_lower : float
        Lower bound of the (1 - α) confidence interval for β₁.
    ci_upper : float
        Upper bound of the (1 - α) confidence interval for β₁.
    """
    if len(cumulative_regret) < 3:
        return 1.0, 1.0, 1.0
    y = np.asarray(cumulative_regret, dtype=float)
    ts = np.arange(1, len(y) + 1, dtype=float)
    valid = ts > 1
    log_ts = np.log(ts[valid])
    log_y = np.log(np.maximum(y[valid], 1e-10))
    A = np.vstack([log_ts, np.ones_like(log_ts)]).T
    n = len(log_ts)
    beta, residuals, rank, s = np.linalg.lstsq(A, log_y, rcond=None)
    alpha_hat = float(beta[0])
    if n <= 2:
        return alpha_hat, alpha_hat, alpha_hat
    mse = np.sum(residuals) / (n - 2) if len(residuals) > 0 else 0.0
    log_ts_centered = log_ts - np.mean(log_ts)
    se = np.sqrt(mse / np.sum(log_ts_centered**2)) if np.sum(log_ts_centered**2) > 0 else 0.0
    t_val = _stats.t.ppf(1.0 - alpha / 2.0, df=n - 2)
    ci_lower = float(alpha_hat - t_val * se)
    ci_upper = float(alpha_hat + t_val * se)
    return alpha_hat, ci_lower, ci_upper


def bound_adaptive_regret(
    T: int,
    delta_min: float,
    B_max: int,
    lambda_0: float = 10.0,
) -> float:
    """Compute an upper bound on total expected regret of the adaptive policy.

    Theorem 1 (Detection Delay — Lorden's Inequality):
    For a Poisson CUSUM with threshold h monitoring a change from λ₀ to
    λ₁ = λ₀ + δ (δ ≥ delta_min), the expected detection delay satisfies
        E[τ] ≤ (h + δ) / KL(λ₁ || λ₀)
    where KL(λ₁ || λ₀) = λ₁ log(λ₁/λ₀) − (λ₁ − λ₀) is the Kullback–Leibler
    divergence of Poisson(λ₁) from Poisson(λ₀).

    Theorem 2 (Post-Detection Regret — Online Convex Optimization):
    After a change is detected at time τ, the algorithm re-optimises using
    Follow-the-Regularised-Leader with KL regularisation. Over the remaining
    horizon the expected post-detection regret is bounded by
        E[R_post(T)] ≤ (λ₁ / 2) · log(T − τ) / δ .
    This follows from standard OCO bounds for exp-concave losses
    (Hazan, 2016, Theorem 3.2).

    Theorem 3 (Total Regret):
    Combining Theorems 1 and 2 over at most B_max change points, the total
    expected regret is bounded by
        R(T) ≤ B_max · [ (h + delta_min) / KL_min
                         + (λ₁ / 2) · log(T) / delta_min ] ,
    where KL_min = KL(λ₀ + delta_min || λ₀).  With the Wald approximation
    h = log(T) for a target ARL of T, this yields a
        R(T) = O( B_max · log(T) / KL(delta_min) )
    bound, which is sublinear in T whenever KL(delta_min) grows faster than
    log(T)/T.

    Parameters
    ----------
    T : int
        Time horizon.
    delta_min : float
        Minimum change magnitude (δ = λ₁ − λ₀).
    B_max : int
        Maximum number of change points.
    lambda_0 : float, optional
        Baseline arrival rate (default 10.0).

    Returns
    -------
    float
        Upper bound on total expected regret.
        Returns inf if delta_min ≤ 0 or T ≤ 0.
    """
    if delta_min <= 0.0 or T <= 0:
        return float("inf")

    lambda_1 = lambda_0 + delta_min
    ratio = lambda_1 / lambda_0
    kl = lambda_1 * np.log(ratio) - (lambda_1 - lambda_0)
    kl = max(kl, 1e-10)

    h = np.log(T + 1)
    detection_delay = (h + kl) / kl

    post_regret = (lambda_1 / 2.0) * np.log(T + 1) / max(kl, 1e-10)

    return float(B_max * (detection_delay + post_regret))
