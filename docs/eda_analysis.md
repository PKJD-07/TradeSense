# TradeSense — EDA & ML Target Definitions

This document describes the exploratory data analysis (EDA) layer and the ML
target definitions for the TradeSense project. It codifies the execution
convention, the mathematical target definitions, the temporal methodology, and
the anti-leakage rules that all downstream ML work must respect.

Scope: this layer performs **data quality assessment, return/volatility/drawdown
analysis, statistics, visualization, and target-label construction only**. It
does **not** implement ML models, feature engineering, trading strategies,
backtesting, or risk management.

---

## 1. Execution / Timestamp Convention (read this first)

Every sample in TradeSense is indexed by a daily session `t`. The convention
that governs which data may flow into features versus targets:

| Event | Allowed data |
|---|---|
| **Features** for sample `t` | Information available at or before the **close of session `t`** |
| **Signal** | Generated **after the close of session `t`** |
| **Execution** | At the **open of session `t+1`** — the earliest actionable moment |
| **Targets** for sample `t` | Information strictly after `close_t`, anchored at the `t+1` open |

> **Why not close-to-close?** A naive target `close_{t+1}/close_t − 1` includes
> the overnight move `close_t → open_{t+1}`, which is *not capturable* under our
> convention — we cannot transact at the same close that generates the signal.
> Anchoring every target at `open_{t+1}` makes the label consistent with the
> execution model. The close-to-close return is still computed in the analysis
> layer as a descriptive statistic, but it is never used as a target.

**Timestamp policy:** all internal timestamps are timezone-aware and normalized
to UTC by the data layer. The analysis layer preserves this; frames converted
from `Candle` objects carry a UTC-aware `DatetimeIndex`.

## 2. Mathematical Target Definitions

All labels are strictly future relative to the feature timestamp `t`.

Let the tradable open-to-close return of a session be

```
r_{t+1}^OC = close_{t+1} / open_{t+1} − 1
```

### 2.1 Primary V1 target — Next-session direction

```
y_t = +1  if r_{t+1}^OC >  ε
y_t = −1  if r_{t+1}^OC < −ε
y_t =  0  if |r_{t+1}^OC| ≤ ε          (neutral zone)
ε = 0.001  (default, = 0.10 %; configurable, NOT tuned on the test set)
```

- **Interpretation:** "does the market move up or down between the next
  session's open and close?" A `+1` means a long position entered at the `t+1`
  open and closed at the `t+1` close makes money.
- **Neutral zone:** rows with `|move| ≤ ε` are labeled `0`. For a V1 binary
  classifier these rows are typically dropped; alternatively they can be kept as
  a third class. The epsilon default is a modeling choice, not a tuned
  hyperparameter — do not optimize it on the test set.
- **Evaluation:** accuracy vs. a 50% baseline (after dropping neutral rows), or
  a long/short confusion analysis. No model is needed to sanity-check the label
  distribution.

### 2.2 Secondary target — N-session forward return

```
y_t = close_{t+N} / open_{t+1} − 1      N = 5 (default, configurable)
```

Entry at the `t+1` open, exit at the `t+N` close. Continuous target capturing
the total tradable return over the next `N` sessions.

### 2.3 Secondary target — N-session future realized volatility

```
R_{t+i} = ln( close_{t+i} / open_{t+i} )           i = 1 .. N
y_t = sqrt( Σ_{i=1..N} R_{t+i}² )
      optionally annualized by sqrt(252/N)
```

Realized volatility of the next `N` **open-to-close log returns**, matching the
execution convention. `N = 5` by default.

## 3. Adjusted vs. Executable Prices

The Yahoo provider is used with `auto_adjust=True` (the default). Adjusted
closes are **restated** so that returns are continuous across dividends and
stock splits.

- Adjusted prices are the correct input for **return computation, feature
  engineering, and backtesting research** — they remove phantom returns.
- Adjusted prices are **NOT necessarily executable market prices.** A published
  close on a split/dividend date will differ from the adjusted close.
- **Future backtesting must distinguish adjusted research prices from execution
  prices.** This distinction is documented here now so it is never silently
  conflated later. The `volume` series is not adjusted by Yahoo in either mode.

## 4. Analysis Modules

| Module | Purpose |
|---|---|
| `convert` | `Candle`/`CandleCollection` → pandas DataFrame bridge; CSV caching under `data/processed/` (gitignored) |
| `quality` | `assess_quality` report: observation count, date range, missing values, duplicate timestamps, missing trading days, zero/negative volume, OHLC violations, extreme moves |
| `returns` | simple / log / N-period returns, open-to-close returns (all causal) |
| `volatility` | annualized, rolling, and realized volatility (causal) |
| `drawdown` | underwater curve, max drawdown, max drawdown duration |
| `statistics` | descriptive stats, autocorrelation (ACF), cross-asset return correlation. No PACF / no statsmodels for V1 |
| `targets` | the labels in §2 (labels only) |
| `split` | chronological 70/15/15 split and walk-forward windows with purge gaps |
| `plots` | price/return/distribution/vol/volume/drawdown/ACF/correlation figures (Agg backend, saved to `figures/`) |

**Configuration** (not hard-coded in the library): the notebook drives the
universe (`AAPL, MSFT, JPM, XOM, SPY`), the date range (`2021-01-01` →
`2026-08-01`), the cache directory (`data/processed/`), and the figure
directory (`figures/`). Both directories are gitignored; generated datasets and
downloaded market data are never committed.

## 5. Temporal Methodology

Financial time series are **not** IID: returns are autocorrelated, volatility
clusters, and N-period forward targets make labels of nearby samples share
future information. Consequences:

- **Random splits are inappropriate.** A random split places "yesterday" in the
  test set and "today" in training, leaking the future into model selection.
- **Chronological split:** train → validation → test in time order
  (default 70/15/15). Use `gap ≥ N` rows between segments to purge
  overlapping labels and rolling-window feature bleed.
- **Walk-forward / rolling-origin validation:** repeatedly fit on a train block
  and evaluate on the immediately following test block, advancing in time. This
  simulates live conditions where the model only sees the past.
- **Causal features:** every feature at row `t` must be computable from rows
  `≤ t` only (no centered/backward-looking statistics).
- **Fit pre-processing on train only:** e.g. a scaler fit on the training block
  must be applied (not refit) to the test block.
- **Purge / embargo:** drop `gap` rows between train and test so labels that
  overlap the boundary (e.g. a forward return spanning the split) are removed
  from training.

## 6. Anti-Leakage Checklist

Before any training run, verify each item:

- [ ] Target at row `t` uses only rows strictly after `t` (see §2 — anchors at
      `open_{t+1}`).
- [ ] No feature at row `t` uses data after `close_t` (no `shift(-k)`, no
      future normalization constants).
- [ ] Scaling / imputation fit **only on the train split**, applied to test.
- [ ] Splits are chronological; purge gap ≥ forward horizon `N`.
- [ ] No model tuning (epsilon, hyperparameters) on the test set — validation
      only.
- [ ] No PACF / statsmodels in V1; only descriptive stats, ACF, correlations.
- [ ] Adjusted prices used for returns; executable vs adjusted never conflated.
- [ ] No live API calls inside automated tests; deterministic synthetic data.

## 7. V1 Recommendation (summary)

The **next-session direction** (label in §2.1) is the V1 primary target because
it is (a) directly actionable — it maps to long/short under the execution
convention, (b) evaluable immediately without a model (accuracy vs. 50%, label
balance), (c) robust to noisy daily returns via the neutral zone, and
(d) minimal — it defers regression and volatility modeling to later phases.
The forward-return and future-realized-volatility labels are provided as
secondary targets for later phases.
