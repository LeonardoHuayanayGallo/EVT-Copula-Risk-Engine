"""
copula_model.py
Implements Gaussian and Student-t Copulas to model
asymmetric tail dependence between assets.

Why Copulas?
- Correlation matrices assume linear, symmetric dependence.
- In financial crises, correlations spike and assets crash TOGETHER.
- Copulas separate marginal distributions from joint dependence structure.
- Student-t Copula has heavier joint tails — captures financial contagion.
- This is what Gaussian Copula CDO models missed in 2008.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def to_uniform_margins(returns: np.ndarray) -> np.ndarray:
    """
    Transform each asset's returns to uniform [0,1] margins
    using the empirical CDF (probability integral transform).
    First step in copula fitting.
    """
    n, d = returns.shape
    u = np.zeros_like(returns, dtype=float)
    for j in range(d):
        u[:, j] = stats.rankdata(returns[:, j]) / (n + 1)
    return u


def fit_gaussian_copula(returns: np.ndarray) -> np.ndarray:
    """
    Fit Gaussian Copula to multivariate returns.
    Returns the rank correlation matrix.

    Gaussian Copula UNDERESTIMATES joint extreme losses
    because it has zero tail dependence (lambda_L = 0).
    """
    u = to_uniform_margins(returns)
    z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    corr = np.corrcoef(z.T)
    print(f"[COPULA] Gaussian Copula fitted — rank correlation matrix computed")
    return corr


def fit_t_copula(returns: np.ndarray, nu_init: float = 5.0):
    """
    Fit Student-t Copula to multivariate returns.
    Estimates degrees of freedom (nu) via MLE.

    Student-t Copula has POSITIVE tail dependence (lambda_L > 0).
    When one asset crashes, others are more likely to crash simultaneously.
    """
    u = to_uniform_margins(returns)
    n, d = u.shape

    def neg_log_likelihood(params):
        nu = float(params[0])
        if nu <= 2.01:
            return 1e10
        try:
            t_scores = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)
            corr = np.corrcoef(t_scores.T)
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -0.999, 0.999)
            sign, log_det = np.linalg.slogdet(corr)
            if sign <= 0:
                return 1e10
            corr_inv = np.linalg.inv(corr)
            ll = 0
            for i in range(min(n, 500)):  # subsample for speed
                x = t_scores[i]
                quad_full = float(x @ corr_inv @ x)
                quad_marg = float(np.sum(x ** 2))
                ll += (
                    -0.5 * log_det
                    - ((nu + d) / 2) * np.log(1 + quad_full / nu)
                    + ((nu + 1) / 2) * np.sum(np.log(1 + x**2 / nu))
                )
            return -ll
        except Exception:
            return 1e10

    result = minimize(
        neg_log_likelihood,
        x0=[nu_init],
        bounds=[(2.1, 50)],
        method="L-BFGS-B"
    )
    nu_hat = float(result.x[0])

    t_scores = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu_hat)
    corr = np.corrcoef(t_scores.T)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    print(f"[COPULA] Student-t Copula fitted")
    print(f"         Degrees of freedom (nu) = {nu_hat:.2f}")
    print(f"         Lower nu = heavier joint tails = more contagion risk")

    return corr, nu_hat


def tail_dependence_coefficient(corr: np.ndarray, nu: float) -> float:
    """
    Lower tail dependence coefficient lambda_L for Student-t Copula.

    This is the probability that asset A crashes GIVEN asset B crashes.
    Formula: lambda_L = 2 x t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))

    For Gaussian Copula: lambda_L = 0 (the fatal assumption)
    For Student-t:       lambda_L > 0 (joint crash probability exists)
    """
    n = corr.shape[0]
    idx = np.triu_indices(n, k=1)
    rho_avg = float(np.mean(corr[idx]))
    rho_avg = np.clip(rho_avg, -0.999, 0.999)

    arg = -np.sqrt((nu + 1) * (1 - rho_avg) / (1 + rho_avg))
    lam = 2 * stats.t.cdf(arg, df=nu + 1)

    print(f"[COPULA] Tail Dependence Coefficient (lambda_L) = {lam:.6f}")
    print(f"         {lam*100:.2f}% probability of joint extreme loss")
    print(f"         Gaussian Copula gives lambda_L = 0.000000")

    return lam