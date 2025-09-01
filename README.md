Machine Learning Event-Driven Trading (Small-Cap Biotech, End-of-Day)

Universe: U.S. biotech, market cap $100M–$2B
Style: Event-driven, daily bars, T+1 execution (no intraday)
Goal: Exploit post-news adjustments and test whether a systematic ML approach can beat passive biotech exposure (XBI).

TL;DR
	•	Always-on ML loses money in this domain.
	•	Regime-aware gating turns a negative expectancy into a positive one.
	•	In the main backtest window, the GATED strategy outperforms XBI on return, drawdown, and Sharpe (net of costs).

⸻

1) Problem Statement

Small-cap biotech is highly event-driven (FDA, clinical results, partnerships). Moves are large, liquidity is thin, coverage is sparse—inefficiencies can persist for days.
This project builds a daily ML pipeline (news/NLP, options, technicals, macro) to generate regime-aware trade signals.
Signals at T are executed T+1 at the open to avoid look-ahead.

⸻

2) Data & Feature Engineering

News @ T
	•	Google News RSS + Finnhub; deduped & normalized titles.
	•	FinRoBERTa sentiment (neg/neu/pos + compound).
	•	FinBERT (OOF-tuned) features: nlp_logit, nlp_margin, nlp_entropy.
	•	Interpretable event flags: FDA, clinical, M&A/licensing, legal.

Options / Technical / Macro @ T−1
	•	Options (Polygon.io): avg call IV, put/call, total & per-contract volumes, IV skew (robust to missing puts).
Robust transforms: log1p (e.g., opt_avg_iv_call_ln), median imputation, constrained forward fill.
	•	Technicals (per ticker): ATR/vol 5–10d, momentum 5–20d, 20d cum-return, 20d max drawdown, volume spikes.
	•	Sector & Macro: IBB/XBI volatility & relative strength, VIX, yield curves (2s10s, 3m–10y), USD (DTWEXBGS), IG/HY OAS, breadth.

Anti-leakage
	•	All series shifted to T−1; merge_asof on a business-day calendar.
	•	Not all features are fully stationarized to preserve regime information; leakage is handled by time-aware CV and the meta-gate.

⸻

3) Methods (leakage-aware)
	•	Validation: Forward, Purged K-Fold + per-ticker Embargo (5d) (per López de Prado).
	•	Model (stacking): XGBoost + Random Forest → Logistic Regression (L2) as meta-learner.
	•	XGB: non-linear interactions; RF: variance reduction; LR: stabilization/calibration of probabilities.
	•	Regime Meta-Gate (soft): a macro/volatility classifier predicts when the base stack is reliable.
Final prob = 0.5 + gate * (base_prob − 0.5) (amplify in good regimes, damp in bad ones).

⸻

4) Backtest Design (deterministic, daily)
	•	Execution: Signal at T → fill T+1 open (fallback close).
	•	Sizing: Equity × risk% / ATR$ stop distance; risk% scales with model confidence.
	•	Exits: target (+7%), stop (−4%), ATR-trail, time exit (5d), and a “hard stop” proxy for gap downs.
	•	Costs: commission 10 bps/side + slippage 40 bps/side ⇒ ~100 bps round-trip (net applied).
	•	Risk: Kill-switch closes everything at −30% peak-to-trough drawdown.
	•	Outputs: equity curve (daily), full trade ledger, missing-data log.

⸻

5) Predictive Skill (OOF, forward-purged with embargo)
	•	Base stacker AUC: ~0.562
	•	Regime-blended AUC: ~0.550
	•	Permutation test: p < 0.001 (non-random ranking skill)
	•	By quarter: strongest in 2025Q2 (~0.60 AUC)

A higher-AUC “shadow” variant (AUC ~0.57–0.58 OOF; ~0.63 in 2025Q2) did not improve P&L—likely probability-calibration issues. Next steps: Platt / Isotonic and more regularized stacking (e.g., ElasticNet).

⸻

6) Backtest Results (net of costs)

Full sample (summary):
	•	BASE (always-on): Sharpe −1.07, MaxDD −30.2%, TotRet −27.8%, expectancy −0.65%/trade.
	•	FINAL (always-on, regime blend): Sharpe −1.72, MaxDD −30.1%, TotRet −27.6%, expectancy −0.32%/trade.
	•	GATED (regime filter): Sharpe +1.36, MaxDD −9.2%, TotRet +10.8%, expectancy +0.98%/trade.
Fewer trades (≈400) and higher quality.

Benchmark comparison (aligned 3-month window, 2025Q2):
	•	GATED vs XBI: TotRet +10.8% vs +0.2%; MaxDD −9.2% vs −18.4%; Sharpe(ann) 1.48 vs 0.20.
	•	Beta to XBI: ~0.26 (low).
	•	Daily alpha ≈ +0.16%; HAC/Newey–West t ≈ 0.85, p ≈ 0.39 (short sample, not statistically significant).

Cost sensitivity (round-trip from 1.0% to 2.2%): returns decay monotonically with slippage; edge approaches zero around ~0.9–1.0% per side.

⸻

7) Limitations
	•	Short horizon (Jan 2024 → Jul 2025) due to free APIs; may miss full cycles.
	•	Options coverage: ~40% imputed → weaker options signal.
	•	NLP: FinRoBERTa + FinBERT features are basic; biotech-specific NLP likely adds edge.
	•	Calibration: raw probabilities (no Platt/Isotonic) → sub-optimal thresholding/sizing.
	•	Execution realism: constant % slippage, no liquidity/borrow constraints, daily bars only.

⸻

8) Roadmap
	•	Probability calibration (Platt/Isotonic), precision@k & PR-AUC targeting.
	•	Liquidity & risk caps (gross/net, per-name, ADV filters, borrow).
	•	Richer options features and smarter missingness.
	•	Biotech-aware NLP (FDA stage/context, trial nuance).
	•	Paper trading (3–6 months) with immutable logs to validate capacity/costs and stabilize thresholds.

⸻

9) Reproducibility (high level)
	•	All features timestamp-aligned (T−1), purged/embargoed CV, fixed seeds, and deterministic backtester.
	•	Key artifacts:
	•	ML.parquet (features + labels)
	•	sent_features_oof.parquet (NLP OOF)
	•	portfolio_equity_GATED.csv, trading_history_GATED.csv (outputs)
	•	Educational content — not investment advice.
