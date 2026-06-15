"""
main.py
Full pipeline:
  1. Download market data via yfinance (free, no API key)
  2. Fit EVT model (GPD via Peaks Over Threshold)
  3. Fit Copulas (Gaussian + Student-t)
  4. Compute VaR, ES, tail dependence coefficient
  5. Generate 5 charts
  6. Print final risk report

Usage:
  python3 main.py
  python3 main.py --tickers SPY QQQ GLD TLT --confidence 0.99
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import data_loader
import evt_model
import copula_model
import risk_metrics
import visualizer


def main():
    parser = argparse.ArgumentParser(description="EVT + Copula Risk Engine")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["SPY", "QQQ", "GLD", "TLT"],
        help="List of tickers (default: SPY QQQ GLD TLT)"
    )
    parser.add_argument("--start",       default="2018-01-01")
    parser.add_argument("--end",         default="2024-12-31")
    parser.add_argument("--confidence",  type=float, default=0.99)
    parser.add_argument("--threshold_q", type=float, default=0.90)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  QUANTITATIVE RISK ENGINE")
    print("  EVT + Copula Portfolio Risk Analysis")
    print("="*60 + "\n")

    # STEP 1: Download data
    returns_df       = data_loader.download_returns(
        args.tickers, start=args.start, end=args.end
    )
    portfolio_returns = data_loader.get_portfolio_returns(returns_df)
    port_array        = portfolio_returns.values

    print(f"\n[PORTFOLIO] {len(args.tickers)} assets, "
          f"equal-weighted, {len(port_array)} observations\n")

    # STEP 2: EVT analysis
    print("─"*40)
    print("STEP 2: Extreme Value Theory")
    print("─"*40)
    evt_results = evt_model.run_evt_analysis(
        port_array,
        confidence=args.confidence,
        threshold_q=args.threshold_q
    )

    # STEP 3: Copula analysis
    print("\n" + "─"*40)
    print("STEP 3: Copula Dependence Modeling")
    print("─"*40)
    returns_array  = returns_df.values
    gaussian_corr  = copula_model.fit_gaussian_copula(returns_array)
    t_corr, nu     = copula_model.fit_t_copula(returns_array)
    tail_dep       = copula_model.tail_dependence_coefficient(t_corr, nu)

    # STEP 4: Risk report
    print("\n" + "─"*40)
    print("STEP 4: Final Risk Report")
    print("─"*40)
    report = risk_metrics.build_risk_report(
        evt_results, args.tickers, tail_dep, nu, portfolio_returns
    )
    print("\n" + report.to_string(index=False))
    report.to_csv("output/risk_report.csv", index=False)
    print("\n[REPORT] Saved to output/risk_report.csv")

    # STEP 5: Charts
    print("\n" + "─"*40)
    print("STEP 5: Generating Charts")
    print("─"*40)
    label = " + ".join(args.tickers)

    visualizer.plot_return_distribution(
        port_array,
        evt_results["evt_var"], evt_results["normal_var"],
        evt_results["hist_var"], args.confidence, label
    )
    visualizer.plot_tail_fit(
        port_array, evt_results["xi"],
        evt_results["beta"], evt_results["threshold"]
    )
    visualizer.plot_risk_comparison(evt_results)
    visualizer.plot_rolling_var(
        portfolio_returns,
        evt_results["evt_var"], evt_results["normal_var"]
    )
    visualizer.plot_correlation_heatmap(
        returns_df, gaussian_corr, t_corr, nu
    )

    # Final summary
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n  Portfolio : {' + '.join(args.tickers)} (equal-weighted)")
    print(f"  Period    : {args.start} to {args.end}")
    print(f"  Confidence: {args.confidence*100:.0f}%\n")
    print(f"  Normal ES  : {evt_results['normal_es']*100:.4f}%")
    print(f"  EVT ES     : {evt_results['evt_es']*100:.4f}%")
    print(f"  Tail Shape : xi = {evt_results['xi']:.4f}")
    print(f"  Tail Dep.  : lambda = {tail_dep:.4f}")
    print(f"\n  Output files saved to /output/")
    print(f"  01_return_distribution.png")
    print(f"  02_gpd_tail_fit.png")
    print(f"  03_risk_comparison.png")
    print(f"  04_rolling_var_drawdown.png")
    print(f"  05_copula_correlation.png")
    print(f"  risk_report.csv\n")


if __name__ == "__main__":
    main()