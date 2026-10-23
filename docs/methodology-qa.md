# Methodology Q&A: the ten hardest questions

These are the questions a quant trader, risk manager or examiner is most likely to press on. Every number comes from `results/` (mainly `summary.json`); section references are to the README.

---

### 1. Why run Engle–Granger in both directions and keep the larger p-value? Is that a valid p-value?

In finite samples the EG test depends on which series is the regressand: the residual from regressing $y$ on $x$ is not the same as the one from regressing $x$ on $y$ (unless $R^2=1$). If I tested both and kept whichever looked better, I would be running two tests at 5% and reporting the minimum, so the true size could approach 10%. Keeping $p=\max(p_{yx},p_{xy})$ removes that choice. It is valid, and conservative: under the null, $P(\max(p_{yx},p_{xy})\le\alpha)\le P(p_{yx}\le\alpha)\le\alpha$. So the p-value is super-uniform, which is all Benjamini–Hochberg needs. (MacKinnon p-values are themselves approximations, so "valid" means approximately valid.) The data show the conservatism. Directional tests reject 8.2% of the time, but only 4.7% of pairs (256 of 5,455) reject in both directions. Johansen is symmetric, but statsmodels gives only critical values and it over-rejects in short samples (12.2%). That is why I use it only as a confirmation filter, and under BH it never binds.

### 2. Why Benjamini–Hochberg rather than Bonferroni, and is BH valid when pairs share a stock?

Bonferroni controls the family-wise error rate, the chance of even one false pair. With about 174 tests it needs $p<0.05/174\approx2.9\times10^{-4}$. For a portfolio of up to 20 pairs, a few false pairs are tolerable. What matters is the *share* of false pairs, which is exactly the false discovery rate, so FDR is the right error rate. Ex post, the choice did not matter: Bonferroni within each window keeps the same 7 pair-windows as BH. BH is proven under independence or positive regression dependence (PRDS). Pairs that share a leg (JPM/BAC and JPM/C) are dependent, and PRDS is not guaranteed for these statistics. Benjamini–Yekutieli is valid under any dependence. It divides $q$ by $H_m=\sum_{i=1}^{m}1/i\approx5.74$ for $m=174$, which makes the effective $q$ about 0.9%, and it keeps 4 pair-windows instead of 7. The conclusion is the same under every correction: almost nothing survives. One caveat: FDR is controlled within each window, not across all 32.

### 3. What do "8.2% directional rejections vs 5% expected" and Storey's $\pi_0\approx0.89$ actually tell you?

There are 10,910 directional EG tests (5,455 pairs × 2). If no pair were cointegrated, about 5% would reject at the 5% level. We see 8.2%, a mild excess. Storey's estimator uses the fact that p-values above 0.5 come almost only from true nulls, which are uniform. Here about 45% of p-values exceed 0.5, so $\hat\pi_0=0.447/0.5\approx0.89$: roughly one test in ten comes from a genuinely cointegrated pair. Combining the two numbers gives the implied power on those pairs: $(0.082-0.89\times0.05)/0.11\approx35\%$. So real relationships exist, but a 252-day test detects only about a third of them. At the pair level, 256 pairs reject in both directions. That is below 273 (5% of 5,455), which is the *upper bound* on false positives if every pair were null; the both-directions rule has size at most 5%. The raw-significant list cannot be told apart from noise. Two caveats. The tests are dependent because legs and factors are shared. And the ADF test can be oversized under heteroskedasticity, so part of the 3.2-point excess may be size distortion rather than cointegration.

### 4. How exactly do you rule out look-ahead bias? What leaks remain?

- **Selection and estimation:** `precompute_tests` hands each window only its 252-day formation slice. The tests, BH, half-life, α, β and spread mean/sd all come from it and are then frozen.
- **Kalman:** the filter is initialised from formation OLS, δ = 10⁻³ is fixed in advance, and day $t$'s z-score uses the prior $\theta_{t|t-1}$, which only uses data up to $t-1$.
- **Execution:** `held[lag:] = desired[:-lag]`, so a signal at the close of $t$ fills at the close of $t+1$. Hedge-ratio weighting uses the β known on the decision day.
- **Capital, diagnostics, parameters:** capital is split by the pair count known at formation; the out-of-sample ADF and z-sd diagnostics never feed back; `StrategyConfig` was fixed before the first run.
- **Remaining leaks:** the universe was chosen in 2026 from survivors, so acquired or delisted firms, the classic permanent pair breaks (Anadarko, BB&T/SunTrust, Xilinx), are missing. That **flatters** the results by removing large stop-losses; a point-in-time universe would most likely look worse. Industry groups are also current, and the ETF pairs are textbook choices. Adjusted closes are not a leak: they are proportional to a total-return index, so taking logs shifts every price by (approximately) the same constant, which α absorbs.

### 5. Under dollar-neutral sizing, does the hedge ratio even matter? What does that do to your Kalman vs OLS comparison?

Each trade is \$1 long and \$1 short, so P&L at entry follows $r_y-r_x$, not $\Delta s_t=\Delta y_t-\beta\,\Delta x_t$. When β ≠ 1 the position also carries a residual $(1-\beta)$ exposure to $x$. β therefore drives only the **signal** (when to trade), not the **position**. The headline Kalman vs OLS comparison is really "static formation spread vs adaptive prediction-error signal". It is not a test of which hedge removes more risk. On that signal comparison there is no significant difference: Kalman minus OLS is +1.5% a year under BH (t = 1.46, p = 0.14) and −0.1% uncorrected (p = 0.96). To test the actual hedge, I re-ran with hedge-ratio weights fixed at entry. The Sharpe ratios are 0.14 (OLS) vs 0.44 (Kalman) under BH, and −0.20 vs −0.00 uncorrected, and none is significant at 5% (the smallest p is 0.059). β can move a lot: for C/PNC it went from 0.90 to 1.5 in March 2020, which would have changed a β-weighted position substantially.

### 6. The Kalman hedge hits far fewer stop-losses. Why isn't that good news?

The model is misspecified. It assumes i.i.d. observation noise, but a cointegrating spread is an AR(1) process with φ close to 1. The filter therefore reads a persistent deviation as news about the state, and the intercept $\alpha_t$ (with β) drifts toward the new prices. The z-score, a scaled one-step prediction error, then decays back to zero even when prices never converge. The exit rule fires and the loss is booked as a "convergence". The evidence:

- The median lag-1 autocorrelation of the Kalman z is 0.90. A well-specified filter's innovations would be near 0.
- Uncorrected, the stop rate is 10% for Kalman against 50% for OLS. But the losses are **relabelled, not avoided**. OLS stopped trades lost 5.88 units of pair capital while its other trades made +5.37. Kalman stops lost only 1.28, but its other trades lost a further 0.14.
- The three worst Kalman trades, INTC/MU in 2024 (−43%) and TXN/MU and ADI/MU in Sep–Oct 2025 (−37% and −31%), all closed on the exit rule.

The fixes: model the spread as a mean-reverting state, estimate δ by maximum likelihood on the formation window, and add P&L-based or time-based stops.

### 7. Out of sample, 92.6% of selected spreads fail to reject a unit root. Doesn't that mean 93% of relationships broke?

No. Failing to reject is weak evidence of a unit root. Three reasons:

- **Low power.** With 126 observations and a half-life of 10–15 trading days (φ ≈ 0.93–0.95), a Dickey–Fuller test typically rejects well under half the time even if the relationship is intact. This is a standard textbook result, not something I estimated here.
- **Selection bias.** Pairs were picked *because* their formation residuals looked most stationary, so their formation half-lives and σ are biased downward (a winner's curse).
- **In-sample fit.** OLS minimises in-sample residual variance, so the out-of-sample sd tends to exceed 1 even under a true relationship.

The better evidence is about magnitudes (uncorrected selection, 175 pair-windows):

- The median out-of-sample z sd is 1.47, and 26% of pair-windows exceed 2.
- On average 19.5% of days sit beyond |z| = 4, against 0.006% under N(0,1).
- Half of OLS trades end in a stop-loss.
- P&L lines up with the diagnostic: the 13 pair-windows that still reject a unit root made +5.3%, while the other 162 lost −0.7% (OLS).

That points to widespread weakening, but I cannot say precisely how many relationships broke.

### 8. BH/Kalman has a Sharpe of 0.42 and an alpha with p = 0.049. Isn't that an edge?

I don't think so. The weaknesses:

- **Thin sample.** The result rests on 16 trades in 6 windows, and 7 of those trades are one pair (PNC/MS, 2018).
- **Concentration.** The two best trades provide 88% of net P&L: EOG/VLO in Jan 2021 (+18%) and C/PNC in Mar 2020 (+12%). The C/PNC gain was a COVID overshoot to z ≈ −6.6, not mean reversion.
- **Significance.** The daily Newey–West p is 0.08, the per-trade t-test gives 0.20 and the per-window t-test 0.17. The Wilcoxon p of 0.03 says the *typical* trade is positive (many small wins), not that the mean is, and trades within a window are not independent.
- **Multiple specifications.** It is the best of four headline specifications, so the alpha p of 0.049 becomes about 0.20 after a Bonferroni adjustment across the four. Kalman also does not beat OLS significantly (p = 0.14).
- **Precision.** The standard error of a Sharpe ratio over 15.8 years is about 0.25, so the 95% interval runs from roughly −0.1 to 0.9.

What would convince me is a fresh out-of-sample period, or a broader universe that produces many more BH pairs.

### 9. What are your returns in excess of, and why Newey–West standard errors?

The returns are on **committed capital**, as in Gatev, Goetzmann and Rouwenhorst. Each selected pair gets an equal share of capital whether or not it trades, and a window with no pairs earns 0. That makes the series conservative and mostly zeros: the BH book holds a position only 5–7% of the time. The book is self-financing. The \$1 long is funded by the \$1 short, the capital sits as collateral earning about the risk-free rate, and I assume the short rebate offsets financing. The returns are therefore already **excess returns**, which is why I compare them with SPY's return *over T-bills* (Sharpe 0.75), not its raw return. For inference I use Newey–West HAC standard errors, because exposure clusters (positions last 11–33 days on average), mean-reverting spread changes are serially correlated, and volatility clusters. The lag rule is $\lfloor4(n/100)^{2/9}\rfloor$ (Newey–West 1994), which gives 9 lags for $n=3{,}975$. Nine lags is short relative to 11–33-day holding periods, so I also report trade-level and window-level tests, which do not lean on thousands of zero-return days. Beta to SPY is −0.01 to −0.03 and insignificant throughout.

### 10. Your sensitivity grid has a cell with Sharpe 0.60. Why isn't that the result? And how much do costs matter?

The grid has 64 cells: 2 selections × 2 hedges × 4 entry thresholds × 4 cost levels. The best cell (BH, Kalman, entry 3.0, **zero** cost) has a Sharpe of 0.60 and p = 0.050, but it rests on 5 trades and returns 0.8% a year. Its Bonferroni p across 64 cells is $\min(1,64\times0.050)=1$, and zero cost is not achievable. Reporting it would be exactly the over-fitting the grid exists to expose. Under the uncorrected selection, all 16 OLS cells are negative *even at zero cost*. Kalman is positive only at entry ≥ 2.5 with costs ≤ 10 bp, peaking at 0.16, and each extra 5 bp removes about 0.06 of Sharpe. On costs, I assume 1 bp commission plus 4 bp spread/slippage, i.e. 5 bp per leg per side, which makes a round trip about 20 bp of pair capital. Borrow is a flat 50 bp a year on the short leg (general collateral). That is reasonable for liquid large caps, optimistic for 2011 and March 2020, and ignores hard-to-borrow fees and recalls. Costs are not what kills the strategy: the uncorrected OLS book loses money even before trading costs (Sharpe −0.13 at zero commission and slippage, with only the 50 bp borrow fee).

---

## Key numbers to remember

| Number | Meaning |
|---|---|
| 4 Jan 2010 – 22 Oct 2026 | Price sample (4,227 days). Trading runs 3 Jan 2011 – 22 Oct 2026 (3,975 days, 15.8 years) |
| 53 tickers, 32 windows | Universe (plus SPY); 252-day formation, 126-day non-overlapping trading windows |
| 5,455 | Pair tests across all windows (153–174 per window) |
| 8.2% vs 5% | Directional EG rejections at 5% vs the null rate; 4.7% (256 pairs) reject both ways |
| π₀ ≈ 0.89 | Storey estimate of the share of true nulls: about 1 test in 10 is non-null |
| 7 / 4 / 175 | Pair-windows kept by BH (in 6 windows) / Benjamini–Yekutieli / uncorrected selection |
| 0.09 / 0.42 | Sharpe, BH/OLS (p = 0.66) / BH/Kalman (p = 0.08), after costs |
| −0.19 / −0.21 | Sharpe, uncorrected OLS (p = 0.42) / Kalman (p = 0.34) |
| 0.75 | SPY Sharpe on excess return over T-bills |
| 16 trades, 88% | BH/Kalman trade count; share of net P&L from its top two trades |
| p = 0.20 / 0.17 / 0.049 | BH/Kalman per-trade t-test / per-window t-test / alpha vs SPY (one of four specs) |
| +1.5% p.a., p = 0.14 | Kalman minus OLS under BH: no significant difference |
| 50% vs 10% | Stop-loss rate, OLS vs Kalman (uncorrected selection) |
| 0.90 | Median lag-1 autocorrelation of the Kalman z (should be ≈ 0) |
| 7.4% | Out-of-sample spreads that still reject a unit root (13 of 175) |
| 1.47 | Median out-of-sample sd of the formation z-score (≈ 1 if stable); 26% above 2 |
| 19.5% | Average share of out-of-sample days beyond the \|z\| = 4 stop (vs 0.006% under N(0,1)) |
| 0.60, 5 trades, p_Bonf = 1 | Best of 64 grid cells (BH/Kalman, entry 3, zero cost): not the result |
| C/PNC +8.5% | Worked example net of costs, OLS (+7.5% Kalman); β drifted from 0.90 to 1.5 |
| 5 bp + 50 bp | Cost per leg per side (≈ 20 bp per round trip); annual short-borrow fee |
| 9 lags | Newey–West lag length, $\lfloor4(n/100)^{2/9}\rfloor$ with n = 3,975 |
