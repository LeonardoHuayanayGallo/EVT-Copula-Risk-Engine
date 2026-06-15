"""
risk_metrics.py
Computes rolling risk metrics and builds the final comparison report.
"""

import numpy as np
import pandas as pd


def compute_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.99
) -> pd.Series:
    """Rolling 1-year Historical VaR."""
    return returns.rolling(window).quantile(1 - confidence) * -1


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Maximum drawdown series."""
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


def build_risk_report(
    evt_results: dict,
    tickers: list,
    tail_dep: float,
    nu: float,
    returns_series: pd.Series
) -> pd.DataFrame:
    """
    Builds the final risk metrics comparison table across three models.
    """
    confidence = evt_results["confidence"]

    report = pd.DataFrame({
        "Metric": [
            f"VaR ({confidence*100:.0f}%)",
            f"Expected Shortfall ({confidence*100:.0f}%)",
            "Tail Shape parameter (xi)",
            "GPD Scale parameter (beta)",
            "Tail Dependence (t-Copula)",
            "Tail Dependence (Gaussian)",
            "t-Copula Degrees of Freedom",
            "ES Improvement vs Normal",
        ],
        "Normal Model": [
            f"{evt_results['normal_var']*100:.4f}%",
            f"{evt_results['normal_es']*100:.4f}%",
            "0.0000 (assumed)",
            "N/A",
            "0.0000 (assumed)",
            "0.0000",
            "Infinity (assumed)",
            "Baseline",
        ],
        "Historical Simulation": [
            f"{evt_results['hist_var']*100:.4f}%",
            f"{evt_results['hist_es']*100:.4f}%",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        ],
        "EVT + Copula Model": [
            f"{evt_results['evt_var']*100:.4f}%",
            f"{evt_results['evt_es']*100:.4f}%",
            f"{evt_results['xi']:.6f}",
            f"{evt_results['beta']:.6f}",
            f"{tail_dep:.6f}",
            "0.000000",
            f"{nu:.2f}",
            f"{evt_results['es_improvement_pct']:.1f}%",
        ],
    })

    return report