"""
visualizer.py
Generates all charts and saves them to output/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm
import os

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "normal":    "#2196F3",
    "hist":      "#4CAF50",
    "evt":       "#F44336",
    "portfolio": "#1F3864",
    "gold":      "#F2C94C",
}


def plot_return_distribution(
    returns, evt_var, normal_var, hist_var,
    confidence, ticker_label, save_dir="output"
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(returns, bins=100, density=True, alpha=0.5,
            color=COLORS["portfolio"], label="Empirical Returns", edgecolor="white")
    mu, sigma = np.mean(returns), np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 500)
    ax.plot(x, norm.pdf(x, mu, sigma), color=COLORS["normal"],
            linewidth=2, label="Normal Distribution (assumed)")
    ax.axvline(-normal_var, color=COLORS["normal"], linestyle="--",
               linewidth=1.5, label=f"Normal VaR {confidence*100:.0f}% = {normal_var*100:.3f}%")
    ax.axvline(-hist_var,   color=COLORS["hist"],   linestyle="--",
               linewidth=1.5, label=f"Historical VaR = {hist_var*100:.3f}%")
    ax.axvline(-evt_var,    color=COLORS["evt"],    linestyle="--",
               linewidth=2.5, label=f"EVT VaR = {evt_var*100:.3f}%")
    ax.set_title(
        f"{ticker_label} — Return Distribution & VaR Comparison\n"
        "EVT captures fat tails that the Normal model ignores",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, "01_return_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_tail_fit(returns, xi, beta, threshold, save_dir="output"):
    losses = -returns
    exceedances = losses[losses > threshold] - threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Empirical vs GPD density
    ax = axes[0]
    ax.hist(exceedances, bins=40, density=True, alpha=0.6,
            color=COLORS["evt"], label="Empirical Exceedances", edgecolor="white")
    x = np.linspace(0, exceedances.max(), 300)
    ax.plot(x, genpareto.pdf(x, xi, scale=beta),
            color=COLORS["portfolio"], linewidth=2.5, label="GPD Fit")
    ax.set_title("GPD Fit to Tail Exceedances\n(Peaks Over Threshold)",
                 fontweight="bold")
    ax.set_xlabel("Exceedance above threshold u")
    ax.set_ylabel("Density")
    ax.legend()
    ax.text(0.97, 0.95, f"xi = {xi:.4f}\nbeta = {beta:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)

    # Right: Q-Q plot
    ax2 = axes[1]
    theoretical_q = genpareto.ppf(
        np.linspace(0.01, 0.99, len(exceedances)), xi, scale=beta
    )
    empirical_q = np.sort(exceedances)
    ax2.scatter(theoretical_q, empirical_q, alpha=0.5,
                color=COLORS["evt"], s=15)
    max_val = max(theoretical_q.max(), empirical_q.max())
    ax2.plot([0, max_val], [0, max_val], color=COLORS["portfolio"],
             linewidth=2, label="Perfect Fit (45 line)")
    ax2.set_title("Q-Q Plot: GPD Fit Validation", fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles (GPD)")
    ax2.set_ylabel("Empirical Quantiles")
    ax2.legend()

    plt.suptitle("Extreme Value Theory — GPD Tail Fit",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "02_gpd_tail_fit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_risk_comparison(evt_results, save_dir="output"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    confidence = evt_results["confidence"]
    models = ["Normal", "Historical", "EVT"]
    colors = [COLORS["normal"], COLORS["hist"], COLORS["evt"]]
    var_vals = [evt_results["normal_var"]*100,
                evt_results["hist_var"]*100,
                evt_results["evt_var"]*100]
    es_vals  = [evt_results["normal_es"]*100,
                evt_results["hist_es"]*100,
                evt_results["evt_es"]*100]

    for ax, vals, title in zip(
        axes,
        [var_vals, es_vals],
        [f"VaR ({confidence*100:.0f}%)", f"Expected Shortfall ({confidence*100:.0f}%)"]
    ):
        bars = ax.bar(models, vals, color=colors, alpha=0.85,
                      edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss (%)")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3)

    plt.suptitle(
        "Risk Model Comparison: Normal vs Historical vs EVT",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "03_risk_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_rolling_var(
    returns, evt_var_static, normal_var_static, save_dir="output"
):
    from risk_metrics import compute_rolling_var, compute_drawdown

    rolling_var = compute_rolling_var(returns, window=252, confidence=0.99)
    drawdown    = compute_drawdown(returns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(rolling_var.index, rolling_var * 100,
             color=COLORS["evt"], linewidth=1.5, label="Rolling 252-day VaR (99%)")
    ax1.axhline(evt_var_static * 100, color=COLORS["evt"],
                linestyle="--", linewidth=1, alpha=0.6,
                label=f"EVT VaR = {evt_var_static*100:.3f}%")
    ax1.axhline(normal_var_static * 100, color=COLORS["normal"],
                linestyle="--", linewidth=1, alpha=0.6,
                label=f"Normal VaR = {normal_var_static*100:.3f}%")
    ax1.set_ylabel("Daily VaR (%)")
    ax1.set_title("Rolling Value at Risk (99%)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.fill_between(rolling_var.index, rolling_var * 100,
                     alpha=0.1, color=COLORS["evt"])

    ax2.fill_between(drawdown.index, drawdown * 100, 0,
                     color=COLORS["portfolio"], alpha=0.4, label="Drawdown")
    ax2.plot(drawdown.index, drawdown * 100,
             color=COLORS["portfolio"], linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Portfolio Drawdown", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.suptitle("Portfolio Risk Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "04_rolling_var_drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_correlation_heatmap(
    returns_df, gaussian_corr, t_corr, nu, save_dir="output"
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tickers = list(returns_df.columns)
    titles  = [
        "Pearson Correlation\n(Linear, symmetric)",
        "Gaussian Copula\n(Rank correlation, lambda_L = 0)",
        f"Student-t Copula\n(Tail dependence, nu = {nu:.1f})",
    ]
    matrices = [returns_df.corr().values, gaussian_corr, t_corr]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, fontsize=9)
        ax.set_yticklabels(tickers, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax.text(j, i, f"{matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(matrix[i,j]) < 0.7 else "white")

    plt.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    plt.suptitle(
        "Dependence Structure: Pearson vs Gaussian vs Student-t Copula",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "05_copula_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {path}")