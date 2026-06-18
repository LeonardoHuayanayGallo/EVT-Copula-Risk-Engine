# Quantitative Risk Engine: Tail-Risk, Expected Shortfall & Extreme Value Theory

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-yfinance%20free-brightgreen)

## Overview

A quantitative risk engine that measures portfolio tail risk without
assuming returns are normally distributed — the critical flaw behind
most risk model failures, including 2008.

Two frameworks implemented:

1. **Extreme Value Theory (EVT)** — fits a Generalized Pareto Distribution
   to empirical tail losses via the Peaks Over Threshold method
2. **Copula Modeling** — fits Gaussian and Student-t Copulas to capture
   asymmetric tail dependence and financial contagion between assets

All data is pulled free via `yfinance`.

---

## The Problem This Solves

Standard VaR assumes returns follow a normal distribution. They do not:

- Financial returns have **fat tails** — extreme losses occur far more
  frequently than the normal distribution predicts
- Assets have **asymmetric dependence** — they appear uncorrelated in
  normal markets but crash together during crises
- This is why Gaussian Copula CDO models failed in 2008, and why
  Basel III replaced VaR with Expected Shortfall in 2016

---

## Output

![Image Alt](https://github.com/LeonardoHuayanayGallo/EVT-Copula-Risk-Engine/blob/42f97d2643ebbc4a91759a6bce8f13aa1f18bdb3/output/01_return_distribution.png)
![Image Alt](https://github.com/LeonardoHuayanayGallo/EVT-Copula-Risk-Engine/blob/d529e17063263a8c49f8a190fbc85c7eb1be08cf/output/02_gpd_tail_fit.png)
![Image Alt](https://github.com/LeonardoHuayanayGallo/EVT-Copula-Risk-Engine/blob/d529e17063263a8c49f8a190fbc85c7eb1be08cf/output/03_risk_comparison.png)
![Image Alt](https://github.com/LeonardoHuayanayGallo/EVT-Copula-Risk-Engine/blob/d529e17063263a8c49f8a190fbc85c7eb1be08cf/output/04_rolling_var_drawdown.png)
![Image Alt](https://github.com/LeonardoHuayanayGallo/EVT-Copula-Risk-Engine/blob/d529e17063263a8c49f8a190fbc85c7eb1be08cf/output/05_copula_correlation.png)
---

## Methodology

**EVT — Peaks Over Threshold:**
- Threshold u at 90th percentile of empirical losses
- GPD fitted to exceedances via MLE
- VaR: `u + (beta/xi) x [((n/Nu)(1-q))^(-xi) - 1]`
- ES: `(VaR + beta - xi x u) / (1 - xi)`

**Copula Modeling:**
- Uniform margins via empirical CDF (probability integral transform)
- Gaussian Copula: rank correlation, zero tail dependence
- Student-t Copula: degrees of freedom via MLE, positive tail dependence
- Tail dependence: `lambda_L = 2 x t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))`

---

## Technologies

`Python` `SciPy` `NumPy` `Pandas` `yfinance` `Matplotlib` `Seaborn`
