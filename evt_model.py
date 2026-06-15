"""
evt_model.py
Implements Extreme Value Theory (EVT) using the
Peaks Over Threshold (POT) method with a Generalized Pareto Distribution (GPD).

Key insight: standard VaR assumes returns are normally distributed.
EVT makes NO distributional assumption — it fits only the tail empirically.
This produces more accurate VaR and Expected Shortfall in extreme scenarios.
"""

import numpy as np
from scipy.stats import genpareto, norm
import warnings
warnings.filterwarnings("ignore")


def select_threshold(returns: np.ndarray, quantile: float = 0.90) -> float:
    """
    Select threshold u as the q-th quantile of losses (positive values).
    Standard practice: 90th-95th percentile of losses.
    """
    losses = -returns
    u = np.quantile(losses, quantile)
    print(f"[EVT] Threshold u = {u:.6f}  ({quantile*100:.0f}th percentile of losses)")
    return u


def fit_gpd(returns: np.ndarray, threshold: float):
    """
    Fit Generalized Pareto Distribution to exceedances above threshold.

    The GPD has two parameters:
    - xi (shape): tail heaviness. xi > 0 means heavy tail (fat tails like finance)
    - beta (scale): spread of the tail

    Returns GPD parameters and the exceedances.
    """
    losses = -returns
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 20:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold. "
            "Lower the threshold quantile."
        )

    print(f"[EVT] Fitting GPD to {len(exceedances)} exceedances above u={threshold:.6f}")

    # Fit GPD using Maximum Likelihood Estimation
    xi, loc, beta = genpareto.fit(exceedances, floc=0)

    print(f"[EVT] GPD fit complete:")
    print(f"      Shape  (xi)  = {xi:.6f}  "
          f"{'[Heavy tail confirmed]' if xi > 0 else '[Thin tail]'}")
    print(f"      Scale  (beta)= {beta:.6f}")
    print(f"      Exceedances  = {len(exceedances)}")

    return xi, beta, exceedances, threshold


def evt_var(
    returns: np.ndarray,
    xi: float,
    beta: float,
    threshold: float,
    confidence: float = 0.99
) -> float:
    """
    Compute EVT-based Value at Risk at given confidence level.

    Formula (POT method):
    VaR_q = u + (beta/xi) x [((n/Nu) x (1-q))^(-xi) - 1]
    """
    losses = -returns
    n  = len(losses)
    Nu = np.sum(losses > threshold)

    if xi == 0:
        var_evt = threshold - beta * np.log((n / Nu) * (1 - confidence))
    else:
        var_evt = threshold + (beta / xi) * (
            ((n / Nu) * (1 - confidence)) ** (-xi) - 1
        )

    return var_evt


def evt_es(
    returns: np.ndarray,
    xi: float,
    beta: float,
    threshold: float,
    confidence: float = 0.99
) -> float:
    """
    Compute EVT-based Expected Shortfall (Conditional VaR).

    ES = average loss GIVEN that loss exceeds VaR.
    This is the metric Basel III requires banks to use (replaced VaR in 2016).

    Formula: ES = (VaR + beta - xi x u) / (1 - xi)
    """
    var = evt_var(returns, xi, beta, threshold, confidence)
    es  = (var + beta - xi * threshold) / (1 - xi)
    return es


def normal_var(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Parametric VaR assuming Normal distribution."""
    mu    = np.mean(returns)
    sigma = np.std(returns)
    return -(mu + norm.ppf(1 - confidence) * sigma)


def normal_es(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Parametric ES assuming Normal distribution."""
    mu    = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - confidence
    return -(mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha)


def historical_var(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical simulation VaR — simple empirical quantile."""
    return -np.quantile(returns, 1 - confidence)


def historical_es(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical simulation ES — mean of losses beyond VaR."""
    var = historical_var(returns, confidence)
    losses = -returns
    tail_losses = losses[losses > var]
    if len(tail_losses) == 0:
        return var
    return tail_losses.mean()


def run_evt_analysis(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_q: float = 0.90
):
    """
    Full EVT analysis pipeline.
    Returns dict with all VaR/ES estimates for comparison.
    """
    threshold = select_threshold(returns, threshold_q)
    xi, beta, exceedances, u = fit_gpd(returns, threshold)

    results = {
        "evt_var":    evt_var(returns, xi, beta, threshold, confidence),
        "evt_es":     evt_es(returns, xi, beta, threshold, confidence),
        "normal_var": normal_var(returns, confidence),
        "normal_es":  normal_es(returns, confidence),
        "hist_var":   historical_var(returns, confidence),
        "hist_es":    historical_es(returns, confidence),
        "xi":         xi,
        "beta":       beta,
        "threshold":  threshold,
        "n_exceed":   len(exceedances),
        "confidence": confidence,
    }

    es_diff = abs(results["normal_es"] - results["evt_es"])
    results["es_improvement_pct"] = (es_diff / results["normal_es"]) * 100

    print(f"\n[EVT] Risk Metrics at {confidence*100:.0f}% Confidence")
    print(f"      Normal VaR   : {results['normal_var']*100:.4f}%")
    print(f"      EVT VaR      : {results['evt_var']*100:.4f}%")
    print(f"      Normal ES    : {results['normal_es']*100:.4f}%")
    print(f"      EVT ES       : {results['evt_es']*100:.4f}%")
    print(f"      ES difference: {es_diff*100:.4f}%")

    return results