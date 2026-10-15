# VIX Futures Basis Backtest

A Python replication of the hedged VIX futures trading strategy in

> Simon, D. P. & Campasano, J. (2014). *The VIX Futures Basis: Evidence and Trading Strategies.* The Journal of Derivatives.

The paper's finding is that the VIX futures basis (futures price minus spot VIX) does not predict changes in the VIX itself, but it does predict VIX futures returns:

- In contango, futures tend to roll down toward spot.
- In backwardation, they tend to roll up toward spot.

The strategy harvests that roll and hedges most of the market risk with E-mini S&P 500 futures.

This repository implements the paper's rules, hedge construction, transaction costs and statistical tests in a single script, `vix_basis_backtest.py`.

---

## Strategy

### Term-structure state
- **Basis** = VIX futures price − spot VIX
- **Contango**: basis > 0 (futures above spot, upward-sloped curve)
- **Backwardation**: basis < 0 (futures below spot, inverted curve)

### Signal
- **Contract**: the front VIX future with at least 10 business days to settlement.
- **Daily roll** = basis ÷ business days to settlement. This is the expected profit per day if the basis decays linearly to settlement.
- **Short** 1 VIX future when in contango with daily roll > +0.10 (≥ $100 expected roll per day).
- **Long** 1 VIX future when in backwardation with daily roll < −0.10.
- *Optional entry filter (paper, Table 5)*: VIX at entry above or below 21 for shorts, 34 for longs.
- *Optional threshold variant (paper, footnote 13)*: ±0.15.

### Hedge
Every VIX position is paired with E-mini S&P 500 futures **in the same direction**: short VIX with short E-minis, long VIX with long E-minis. VIX futures move inversely to equities, so the E-mini leg offsets the VIX leg's market exposure.

The hedge ratio comes from the paper's equations 3 and 4:

```
ΔF_t = β0 + β1·SPRET_t + β2·(SPRET_t · TTS_t) + ε_t                 (3)

HR_t = |β1·1000 + β2·TTS_{t−1}·1000| / (0.01 · ES_{t−1} · 50)         (4)
```

- **Variables in equation 3:**
  - ΔF is the daily change, in VIX points, of the contract the strategy would trade.
  - SPRET is the E-mini return in percent.
  - TTS is business days to settlement.
- **The coefficients are estimated out of sample on an expanding annual window.** The 2006 fit hedges 2007 trades, the 2006–07 fit hedges 2008 trades, and so on.
- **What HR means:** the number of E-minis whose expected dollar P&L offsets one VIX contract for a 1% S&P move.
  - It is set at entry and **never rebalanced**.
  - Fractional contracts are allowed, as in the paper.

### Exits
| Mode | Rule (the 5-day limit always applies) |
|---|---|
| `time` (baseline) | Exit at the end of 5 business days |
| `roll` | Exit early when the reason for the trade is gone: short if roll < 0.05, long if roll > −0.05 |
| `stop` | Exit early if the hedged loss, net of costs, exceeds $500 |
| `take` | Exit early if the hedged gain, net of costs, exceeds $500 |

No new trade is opened on an exit day. The next trade opens at the end of the next day the entry conditions trigger.

### Transaction costs
- Full quoted VIX futures bid-ask on entry and exit (sell at the bid, buy at the ask)
- ½ E-mini tick ($6.25) per side per contract
- $3 round-trip brokerage per contract, both legs

---

## Output

For short and long trades separately, the script reports four P&L series:

| Series | Meaning |
|---|---|
| **Hedged P&L** | What the strategy earns: VIX leg + E-mini leg, net of costs |
| **Unhedged P&L** | The VIX leg of the *same* trades (an attribution, not a separate strategy) |
| **S&P hedge P&L** | The E-mini leg |
| **Roll P&L** | Daily roll accumulated while trades are open (gross) |

**Statistics for each series:** mean, bootstrap p-value, semi-standard deviation, Sortino ratio, 10%/90% fractiles and winners/losers.

**The p-value** is the share of 10,000 samples of randomly timed trades whose mean beats the actual mean. Each sample has the same number of trades and the same average holding period as the strategy.

**Also printed:** the fitted hedge model per year, the average hedge ratio and the correlation between the two legs.

---

## Usage

Requires Python 3.9+ with `numpy` and `pandas`; `matplotlib` is only needed for `--plot`.

```bash
pip install -r requirements.txt

# Check that the code runs (synthetic data, see below)
python vix_basis_backtest.py --demo

# Run all four exit modes on your data
python vix_basis_backtest.py --spot spot.csv --futures futures.csv

# One configuration, with trade log and chart
python vix_basis_backtest.py --spot spot.csv --futures futures.csv \
    --exit-mode roll --vix-filter above --entry-threshold 0.10 \
    --trades-out trades.csv --plot pnl.png
```

| Option | Values | Default |
|---|---|---|
| `--exit-mode` | `time`, `roll`, `stop`, `take`, `all` | `all` |
| `--vix-filter` | `none`, `above`, `below` | `none` |
| `--entry-threshold` | daily roll in VIX points | `0.10` |
| `--draws` | bootstrap samples | `10000` |
| `--trades-out` | CSV path for the trade log | – |
| `--plot` | PNG path for cumulative P&L | – |

> **About `--demo`:** it generates synthetic prices to show the pipeline runs end to end. Its P&L figures are **meaningless** and say nothing about whether the strategy works.

---

## Data

No market data is included. Historical VIX and E-mini prices are generally licensed and may not be redistributed.

The script expects two CSV files:

**`spot.csv`**
| column | description |
|---|---|
| `date` | trading date |
| `vix` | spot VIX at the observation time |
| `es` | front, rollover-adjusted E-mini S&P 500 futures price at the same time |

**`futures.csv`** (one row per date per listed contract)
| column | description |
|---|---|
| `date` | trading date |
| `expiry` | final settlement date of the contract |
| `bid` | VIX futures bid |
| `ask` | VIX futures ask |

**Sampling.** The paper observes all prices at the same moment: the first minute after 15:00 CST in which the VIX futures bid-ask is ≤ 0.10, or else the close. Daily closing prices work but are less accurate, because spot VIX and VIX futures stop trading at different times.

**Sources.**
- **VIX and VIX futures:** Cboe publishes daily VIX index history and historical VIX futures settlement prices. Settlement data has no bid-ask quotes, so you would need to set `bid`/`ask` around the settlement price, for example ±0.025.
- **E-mini:** history is available from most futures data vendors. SPX or SPY can serve as a rough proxy, at the cost of a slightly different hedge.

---

## Paper results for comparison

Baseline (5-day exits, ±0.10 threshold), January 2007 – December 2011, per contract after costs (paper, Table 4):

| | Hedged mean | Hedged Sortino | Unhedged mean | Unhedged Sortino | Trades |
|---|---|---|---|---|---|
| Short (contango) | $539 | 0.75 | $656 | 0.59 | 82 |
| Long (backwardation) | $908 | 0.53 | $1,040 | 0.42 | 42 |

With the roll-decay exit (Tables 6–7), the hedged Sortino ratio rises to 1.08 for shorts and 0.69 for longs.

---

## Differences from the paper

- **Holidays** are inferred from weekdays missing in `spot.csv`, so business-day counts depend on your data covering every trading day.
- **Roll P&L** is reported gross; the paper spreads transaction costs across all P&L categories.
- **Contract splicing** for the hedge regression uses the same contract on both days of each change. The paper only says changes are "adjusted for breaks between contracts."
- **A position still open on the last day of data** is not counted; the script prints a note instead.
- **The paper's sample** (2006–2011) predates later volatility shocks such as February 2018 and March 2020. Results on later data may differ materially.

---

## Disclaimer

This is an independent replication for research and education. It is not investment advice, and backtested results do not guarantee future performance. Short-volatility strategies can suffer large, sudden losses.

## License

MIT, see `LICENSE`.
