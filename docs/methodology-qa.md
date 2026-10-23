# Methodology Q&A: the ten hardest questions

*Sam Berman, University College Dublin. These are concise answers to the questions an interviewer or examiner is most likely to press on. Every number comes from [`results/`](../results/), produced by `python -m src.pipeline` on data to 2026-10-21.*

---

### 1. Why profile least squares on a grid rather than just optimising all NSS parameters?
NSS is linear in the four betas once τ1 and τ2 are fixed. So for each candidate (τ1, τ2) the betas are exact OLS, and the only search is over two parameters.
- Direct non-linear optimisation of all six parameters often stops at local minima. When τ1 ≈ τ2 the two curvature loadings are almost collinear, and the betas can explode in offsetting pairs.
- The grid guarantees the global optimum *on the grid* and is reproducible. Because it is vectorised, it fits all 6,308 days in about one second.
- The cost is resolution: grid steps are about 10% in τ. The extra fit a continuous optimiser could find is far smaller than the LOO error differences between models.

### 2. NSS has the lowest RMSE. Isn't that just because it has more parameters?
In sample, partly yes. That is why I report a leave-one-maturity-out (LOO) test: drop one maturity, refit, predict it.
- On LOO, NSS still wins: 5.8bp, against 6.5bp for NS with free τ and 11.3bp for Diebold–Li NS. So the extra hump captures real structure, mostly at the 6M–5Y points.
- The wide-τ-grid NSS shows what overfitting looks like. Its in-sample RMSE is lower (2.6bp), but its LOO error is higher (6.9bp), and 11% of days have exploding betas.

### 3. How do the NS parameters map to level, slope and curvature, and why does β0 correlate only 0.60 with the Diebold–Li level proxy?
- β0 is the yield at infinite maturity (level).
- β0 + β1 is the instantaneous short rate, so −β1 is long minus short (slope).
- β2 loads on a hump that peaks at about 1.8τ (curvature). With Diebold–Li τ = 1.37y, that peak is at about 30 months.

With fixed τ, the correlations are:
- −β1 with 10Y−3M: 0.996
- β2 with 2×2Y−3M−10Y: 0.98
- β0+β1 with 3M: 0.997

β0 is the *long-end* asymptote, so it correlates 0.97 with the 20Y. The Diebold–Li proxy (3M+2Y+10Y)/3 includes the short rate, which sat at zero for years while long yields moved. That is why that correlation is only 0.60.

### 4. Why exclude the 1-month bill and the 30-year from the PCA? Isn't that cherry-picking?
- **1M:** its daily changes are dominated by bill supply, debt-ceiling episodes and month-end effects. With the 1M included, PC2 becomes mostly a 1M factor (loading −0.76) rather than a curve slope.
- **30Y:** it has no market data from 2002-02 to 2006-02. FRED's values there are Treasury extrapolations, with +17bp and −16bp jumps at the edges.

Both are reported as robustness checks. Adding the 1M gives 92.7% for three PCs; using all 11 maturities from 2006 gives 92.0%, against 95.9% in the main specification. The conclusion that three factors dominate does not depend on the exclusion.

### 5. Is the PCA factor structure stable?
**The variance share is stable; the factor shapes are not.**
- Three PCs explain 94.7–96.5% of variance in every policy regime.
- Out of sample, loadings estimated on 2001–13 explain 95.5% of 2014–26 changes, against 95.7% for the 2014–26 sample's own loadings.
- At the zero lower bound (2008–15), however, the front end was pinned. PC2's congruence with the full-sample slope factor fell to 0.52, and PC3's fell to 0.24, as curvature moved out to 5–7Y.

The practical point: a PCA hedge estimated over a mixed sample would have had the wrong short-end exposures at the ZLB.

### 6. How do you make sure the recession probabilities are genuinely out of sample?
1. **Expanding window.** At origin month t, the probit is re-estimated from scratch.
2. **Publication lag.** Training pairs (s_τ, USREC_{τ+12}) must satisfy τ + 12 ≤ t − 12. The extra 12 months is an assumed lag before NBER announces turning points (historically 4–21 months). Without it, the model would learn from recession dates that were not yet known at time t.
3. **Inputs.** The spread is the monthly average of daily data, available at the end of month t. The current incomplete month is dropped.
4. **Test.** The unit tests flip every USREC value after a cut-off date and confirm that forecasts at earlier origins do not change.

### 7. The 10y–3m AUC is 0.76 but the Brier skill score is negative. Which is right?
Both are. They measure different things.
- **AUC measures ranking.** Recession months received higher probabilities than non-recession months. The 95% block-bootstrap CI is 0.58–0.95, so it excludes 0.5.
- **Brier measures calibrated probability accuracy.** Over 2022–24 the model assigned up to 78% to recessions that did not happen (22 months above 50%). Those large squared errors push the Brier score (0.086) above the base rate's (0.078).
- Excluding origins after 2021, the Brier skill is +0.20. Even so, the Diebold–Mariano test cannot reject equal accuracy with the base rate in either window (p = 0.69 and 0.22).

The honest summary: the spread carries real ranking information, but with only about six recessions it does not deliver statistically better probability forecasts.

### 8. Why did the 2022–24 inversion not produce a recession (so far)?
I can describe what happened but cannot identify the cause with this data.
- **What happened:** the 10y–3m spread was inverted for 25 consecutive months, down to −1.73pp. That is the deepest inversion in the post-1981 sample.
- **Candidate explanations:**
  - QE-compressed term premia, which make a given spread less informative.
  - Private-sector debt locked in at low fixed rates before the hikes, which blunted their transmission.
  - Large fiscal deficits.
  - An inversion driven by expected *disinflationary* cuts rather than expected recession.
- **The signal may be late, not wrong.** Lags have ranged from about 6 to 24 months, and recent NBER labels are provisional.
- **The model adapts.** As the episode enters the expanding window, the slope coefficient flattens from −0.76 to −0.45, so the model is learning that inversions have become less reliable.

### 9. Explain DV01 neutrality and how you approximate the P&L.
**DV01** is the dollar change in value for a 1bp fall in yield: N × MD × 10⁻⁴. On 2026-10-21 it was \$188.5 per \$1mm for the 2y and \$774.7 per \$1mm for the 10y.
- **Neutral sizing:** set N₁₀/N₂ = DV01₂/DV01₁₀ ≈ 0.24, which is \$53mm of 2s against \$13mm of 10s for \$10k/bp per leg.
- **What that buys:** a parallel shift makes about zero P&L, and a 1bp steepening makes \$10k.

**Daily P&L per leg** has three parts:
- carry, (y − funding) × dt, with funding proxied by the 3M yield;
- roll-down along today's curve, MD × slope × dt;
- price change, −MD·Δy + ½·C·Δy².

On 99% of days, duration + convexity is within \$16 per \$10k DV01 of exact repricing; duration alone misses by up to \$2,180. A DV01-neutral steepener is short convexity, because the 10y has more convexity per DV01. That cost \$32k a year in the backtest.

### 10. Your trades lose money. What did you learn, and how do you know you didn't overfit?
**Overfitting controls:**
- Parameters were chosen on 1981–2001 only, by training Sharpe.
- The 2001–26 test period was never used for selection.
- The grids are small (16 and 4 combinations) and published in full.
- Costs are charged, with a 0–2bp sensitivity.

**Findings:**
1. **Mean reversion lost money in every training configuration** (Sharpe between −0.24 and −0.53) and lost out of sample too (Sharpe −0.32, p = 0.09). Over weeks to months, 2s10s trends with the policy cycle rather than reverting.
2. **The carry signal's strong training Sharpe (1.09) was mostly beta to a steep, falling-rate curve.** The always-steepener had a 1.17 Sharpe over the same period. Out of sample, carry has a small gross edge (Sharpe 0.21, not significant), and its turnover (17 trades a year) means costs of 0.25–0.5bp/leg erase it.

A clean negative result, with the reason understood, is more credible than a backtest with a Sharpe of 2.

---

## Key numbers to remember

| Topic | Number |
|---|---|
| Sample | 2001-07-31 → 2026-10-21, 6,308 days, 11 maturities (30Y masked 994 days) |
| NSS fit | 3.2bp mean RMSE in sample; 5.8bp LOO out of sample |
| NS fit | 5.6bp (free τ) / 10.4bp (Diebold–Li fixed τ = 1.37y) RMSE |
| PCA | PC1 78.0%, PC2 12.6%, PC3 5.3% → **95.9%**; OOS 95.5% |
| ZLB instability | PC2 / PC3 congruence 0.52 / 0.24 in 2008–15 |
| Probit (in-sample, 10y–3m) | b = −0.43, HAC p = 0.006 |
| OOS recession | AUC **0.76** (10y–3m), 0.70 (10y–2y); Brier skill −0.11 (+0.20 pre-2022) |
| 2022–24 | 25 months inverted, min −1.73pp; peak OOS probability 78% (May 2023) |
| Latest | P(recession by Sep 2027) = 8.9% (10y–3m) |
| DV01 | \$188.5 (2y) and \$774.7 (10y) per \$1mm; hedge ratio 0.24 |
| Trades (OOS net Sharpe) | mean reversion −0.32, carry −0.14, always-steepener 0.00 |
| Costs | 0.5bp/leg base case; carry's gross Sharpe 0.21 gone above 0.25bp |
