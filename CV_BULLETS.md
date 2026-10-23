# CV bullets: cointegration pairs trading on US large caps (2011–2026)

- Built a Python walk-forward backtest of cointegration pairs trading on 47 US large caps and 6 ETFs (32 windows, 2011–2026), with a one-day execution lag, 5 bp/leg costs and borrow fees.
- Applied Benjamini–Hochberg FDR control to 5,455 Engle–Granger tests, cutting 175 selected pair-windows to 7, and showed the uncorrected strategy lost money after costs (Sharpe −0.19).
- Diagnosed why a Kalman-filter hedge cut stop-losses from 50% to 10% of trades yet lost up to 43% of pair capital on single trades: misspecified noise let the equilibrium drift with prices.

Repo link: https://github.com/samberman-hi/yield-curve-modelling
