#  Machine Learning Event-Driven Trading Framework (Small-Cap Biotech, End-of-Day)

**Universe:** U.S. biotech, market cap $100M–$2B  
**Style:** Event-driven, daily bars, **T+1 execution** (no intraday)  
**Goal:** Exploit post-news adjustments and test whether a systematic ML approach can beat passive biotech exposure (**XBI**).

> **TL;DR**  
> - Always-on ML loses money in this domain  
> - Regime-aware gating turns a negative expectancy into a positive one  
> - In the main backtest window, the **GATED** strategy outperforms XBI (return, drawdown, Sharpe, net of costs)

---

## 1) Problem Statement

Small-cap biotech is highly event-driven (FDA, clinical results, partnerships).  
Moves are large, liquidity is thin, coverage is sparse → inefficiencies can persist for days.  

This project builds a **daily ML pipeline** (news/NLP, options, technicals, macro) to generate regime-aware trade signals.  
Signals at T are executed **T+1 at the open** to avoid look-ahead.

---

## 2) Data & Feature Engineering

**News @ T**
- Google News RSS + Finnhub; deduped & normalized titles  
- FinRoBERTa sentiment (neg/neu/pos + compound)  
- FinBERT (OOF-tuned): `nlp_logit`, `nlp_margin`, `nlp_entropy`  
- Event flags: FDA, clinical, M&A/licensing, legal  

**Options / Technical / Macro @ T−1**
- Options (Polygon.io): avg call IV, put/call, total & per-contract volumes, IV skew  
  - Robust transforms: log1p (e.g., `opt_avg_iv_call_ln`), median imputation, constrained ffill  
- Technicals (per ticker): ATR/vol 5–10d, momentum 5–20d, 20d cum-return, 20d max drawdown, volume spikes  
- Sector & Macro: IBB/XBI volatility & relative strength, VIX, yield curves (2s10s, 3m–10y), USD (DTWEXBGS), IG/HY OAS, breadth  

**Anti-leakage**
- All series shifted to T−1  
- `merge_asof` on business-day calendar  
- Not all features fully stationarized (to preserve regime info)  
- Leakage handled by time-aware CV + meta-gate  

---

## 3) Methods (Leakage-Aware)

- Validation: Forward **Purged K-Fold + per-ticker Embargo (5d)** (ispired: Marcos López de Prado)  
- Model: **Stacking** → XGBoost + Random Forest → Logistic Regression (L2)  
  - XGB: non-linear interactions  
  - RF: variance reduction  
  - LR: stabilization/calibration of probabilities  
- Regime Meta-Gate (soft): macro/volatility classifier predicts when base stack is reliable  
  - `Final prob = 0.5 + gate * (base_prob − 0.5)`

---

## 4) Backtest Design (Deterministic, Daily)

- **Execution:** Signal at T → fill T+1 open (fallback close)  
- **Sizing:** Equity × risk% ÷ ATR$ stop distance; risk% scales with model confidence  
- **Exits:** +7% target, −4% stop, ATR-trail, 5d max hold, hard-stop proxy  
- **Costs:** commission 10 bps/side + slippage 40 bps/side ≈ 100 bps round-trip (net)  
- **Risk:** Kill-switch closes everything at −30% peak-to-trough drawdown  
- **Outputs:** equity curve (daily), trade ledger, missing-data log  

---

## 5) Predictive Skill (OOF, Forward-Purged with Embargo)

- Base stacker AUC ≈ **0.562**  
- Regime-blended AUC ≈ **0.550**  
- Permutation test: **p < 0.001** (non-random skill)  
- By quarter: strongest in **2025Q2 (~0.60 AUC)**  

A higher-AUC “shadow” variant (~0.57–0.58 OOF; ~0.63 in 2025Q2) did **not** improve P&L → likely probability calibration issues.  
Next: Platt / Isotonic scaling, ElasticNet stacking.

---

## 6) Backtest Results (Net of Costs)

**Full sample**
- BASE (always-on): Sharpe −1.07, MaxDD −30.2%, TotRet −27.8%, expectancy −0.65%/trade  
- FINAL (blend): Sharpe −1.72, MaxDD −30.1%, TotRet −27.6%, expectancy −0.32%/trade  
- GATED (regime filter): Sharpe **+1.36**, MaxDD −9.2%, TotRet **+10.8%**, expectancy **+0.98%/trade**  
→ Fewer trades (≈400), higher quality  

**Benchmark comparison (aligned 2025Q2, 3 months)**
- TotRet: **+10.8% vs +0.2% (XBI)**  
- MaxDD: −9.2% vs −18.4%  
- Sharpe(ann): 1.48 vs 0.20  
- Beta to XBI: ~0.26 (low)  
- Daily alpha ≈ +0.16% (NS, t≈0.85, p≈0.39)  

**Cost sensitivity**  
Round-trip costs from **1.0% → 2.2%**: returns decay monotonically; edge ≈ 0 around 0.9–1.0% per side.

---

## 7) Limitations

- Short horizon (Jan 2024 – Jul 2025) due to free APIs  
- Options coverage: ~40% imputed → weaker signal  
- NLP: basic FinRoBERTa + FinBERT; biotech-specific NLP needed  
- Calibration: raw probabilities (no Platt/Isotonic) → sub-optimal thresholding  
- Execution realism: constant % slippage, no liquidity/borrow constraints, daily bars only  

---

## 8) Roadmap

- Probability calibration (Platt/Isotonic), precision@k & PR-AUC  
- Liquidity & risk caps (gross/net, per-name, ADV filters, borrow)  
- Richer options features + smarter missingness  
- Biotech-aware NLP (FDA stage/context, trial nuance)  
- **Paper trading (3–6m)** with immutable logs for validation  

---

Repo pointers

	- Executive details: 00_executive_module.ipynb
 
	- ML Modeling & CV: 07_ml_experimentation.ipynb
 
	- Backtest engine & stress tests: 08_backtest.ipynb, 09_results_and_limitations.ipynb
---

 **Educational content — not investment advice**
