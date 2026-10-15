#!/usr/bin/env python3
"""
vix_basis_backtest.py

Backtest of the hedged VIX-futures roll strategy in
    Simon, D. P. & Campasano, J. (2014), "The VIX Futures Basis: Evidence and
    Trading Strategies", Journal of Derivatives.

Strategy (as specified in the paper)
------------------------------------
* Contract: front VIX future with >= 10 business days to settlement.
* State:    basis = F - VIX;  > 0 contango, < 0 backwardation.
* Signal:   daily roll = (F - VIX) / business days to settlement.
            SHORT 1 VIX future if roll > +0.10, LONG 1 if roll < -0.10.
* Optional entry filter: VIX at entry above/below 21 (shorts) or 34 (longs).
* Hedge:    always on. E-mini S&P futures in the SAME direction as the VIX leg.
            HR = |(b1*1000 + b2*TTS_{t-1}*1000) / (0.01 * ES_{t-1} * 50)|,
            with (b1, b2) from  dF_t = b0 + b1*SPRET_t + b2*(SPRET_t*TTS_t) + e
            estimated out of sample on an expanding annual window.
            HR is fixed at entry and never rebalanced.
* Exits:    baseline: end of 5 business days. Optional risk-management modes
            (each tested separately, 5-day limit always applies):
              roll : short exits if roll < 0.05, long exits if roll > -0.05
              stop : hedged loss net of costs > $500
              take : hedged gain net of costs > $500
* Re-entry: no entry on an exit day; next entry at the end of the next day the
            entry conditions trigger.
* Costs:    full quoted VIX bid-ask on entry and exit, $3 round-trip VIX
            brokerage, 1/2 E-mini tick ($6.25) per side + $3 round-trip per
            E-mini contract. Fractional E-mini contracts allowed (as in paper).
* Stats:    mean, bootstrap p-value vs random entries (same count, same rounded
            average duration), semi-std, Sortino, 10/90% fractiles,
            winners/losers - for hedged, unhedged (VIX leg), hedge leg and roll.

Input data (CSV)
----------------
  spot.csv     columns: date, vix, es
               vix: spot VIX at the observation time
               es : front (rollover-adjusted) E-mini S&P 500 price, same time
  futures.csv  columns: date, expiry, bid, ask
               one row per date and listed VIX futures contract;
               expiry = final settlement date of the contract
  The paper samples all prices synchronously at the first minute after
  15:00 CST in which the VIX-futures bid-ask is <= 0.10 (else the close).
  Daily closes also work but are less accurate, because spot VIX and VIX
  futures stop trading at different times.

Usage
-----
  python vix_basis_backtest.py --demo
  python vix_basis_backtest.py --spot spot.csv --futures futures.csv --exit-mode all
  python vix_basis_backtest.py --spot spot.csv --futures futures.csv \
         --exit-mode roll --vix-filter above --trades-out trades.csv --plot pnl.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    entry_threshold: float = 0.10      # daily roll, VIX points ($100/day)
    exit_roll: float = 0.05            # roll-decay exit threshold
    min_tts: int = 10                  # min business days to settlement
    max_hold: int = 5                  # business days
    exit_mode: str = "time"            # time | roll | stop | take
    stop_loss: float = 500.0           # $ hedged, net of costs
    take_gain: float = 500.0           # $ hedged, net of costs
    vix_filter: str = "none"           # none | above | below
    short_vix_median: float = 21.0
    long_vix_median: float = 34.0
    vix_mult: float = 1000.0           # $ per VIX futures point
    es_mult: float = 50.0              # $ per E-mini point
    vix_brokerage_rt: float = 3.0      # $ round trip per VIX contract
    es_half_tick_usd: float = 6.25     # $ per side per E-mini contract
    es_brokerage_rt: float = 3.0       # $ round trip per E-mini contract
    min_fit_obs: int = 120             # min obs before a hedge model is fitted
    bootstrap_draws: int = 10_000
    seed: int = 42


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
class Market:
    def __init__(self, spot: pd.DataFrame, fut: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        spot = spot.copy()
        spot["date"] = pd.to_datetime(spot["date"])
        spot = spot.sort_values("date").drop_duplicates("date").set_index("date")
        self.dates = spot.index
        self.d64 = self.dates.values.astype("datetime64[D]")
        self.vix = spot["vix"].to_numpy(float)
        self.es = spot["es"].to_numpy(float)

        # Holidays = weekdays inside the sample that have no data row.
        all_bd = pd.bdate_range(self.dates[0], self.dates[-1])
        self.holidays = all_bd.difference(self.dates).values.astype("datetime64[D]")

        fut = fut.copy()
        fut["date"] = pd.to_datetime(fut["date"])
        fut["expiry"] = pd.to_datetime(fut["expiry"])
        idx = {d: i for i, d in enumerate(self.dates)}
        fut = fut[fut["date"].isin(idx.keys())]

        self.quotes: dict[tuple[int, pd.Timestamp], tuple[float, float]] = {}
        self.listed: list[list[pd.Timestamp]] = [[] for _ in self.dates]
        for d, e, b, a in fut[["date", "expiry", "bid", "ask"]].itertuples(index=False):
            i = idx[d]
            if e <= d or not (np.isfinite(b) and np.isfinite(a)):
                continue
            self.quotes[(i, e)] = (float(b), float(a))
            self.listed[i].append(e)
        for lst in self.listed:
            lst.sort()

        self._tts_cache: dict[tuple[int, pd.Timestamp], int] = {}
        self.selected = [self._select(i) for i in range(len(self.dates))]

    def tts(self, i: int, expiry: pd.Timestamp) -> int:
        """Business days from date i until settlement."""
        key = (i, expiry)
        if key not in self._tts_cache:
            self._tts_cache[key] = int(np.busday_count(
                self.d64[i], np.datetime64(expiry, "D"), holidays=self.holidays))
        return self._tts_cache[key]

    def quote(self, i: int, k) -> tuple[float, float] | None:
        return self.quotes.get((i, k))

    def mid(self, i: int, k) -> float:
        b, a = self.quotes[(i, k)]
        return 0.5 * (b + a)

    def _select(self, i: int):
        """Front contract with at least min_tts business days to settlement."""
        for e in self.listed[i]:
            if self.tts(i, e) >= self.cfg.min_tts:
                return e
        return None

    # ---- Determine_VIX_Term_Structure_State --------------------------------
    def term_structure_state(self, i: int, k):
        basis = self.mid(i, k) - self.vix[i]
        state = "CONTANGO" if basis > 0 else "BACKWARDATION" if basis < 0 else "FLAT"
        roll = basis / self.tts(i, k)
        return state, basis, roll

    def daily_roll(self, i: int, k) -> float:
        return self.term_structure_state(i, k)[2]


# --------------------------------------------------------------------------
# Hedge model (equations 3 and 4)
# --------------------------------------------------------------------------
class HedgeModel:
    def __init__(self, mkt: Market):
        self.mkt = mkt
        rows = []
        for i in range(1, len(mkt.dates)):
            k = mkt.selected[i - 1]          # same contract on both days:
            if k is None:                    # no break at contract switches
                continue
            if mkt.quote(i - 1, k) is None or mkt.quote(i, k) is None:
                continue
            d_f = mkt.mid(i, k) - mkt.mid(i - 1, k)
            spret = 100.0 * (mkt.es[i] / mkt.es[i - 1] - 1.0)   # percent
            rows.append((mkt.dates[i].year, d_f, spret, mkt.tts(i, k)))
        self.data = pd.DataFrame(rows, columns=["year", "dF", "spret", "tts"])

        self.coefs: dict[int, np.ndarray] = {}
        self.fit_info = []
        for y in sorted({d.year for d in mkt.dates}):
            sub = self.data[self.data["year"] < y]        # expanding window
            if len(sub) < mkt.cfg.min_fit_obs:
                continue
            X = np.column_stack([np.ones(len(sub)), sub["spret"],
                                 sub["spret"] * sub["tts"]])
            yv = sub["dF"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
            n, p = X.shape
            r2 = 1 - resid.var() / yv.var()
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - p)
            self.coefs[y] = beta
            self.fit_info.append(dict(trade_year=y, fit_through=y - 1, nobs=n,
                                      b0=beta[0], b1=beta[1], b2=beta[2],
                                      adj_r2=r2_adj,
                                      sens_at_10d=beta[1] + 10 * beta[2]))

    def ratio(self, i: int, k) -> float | None:
        """E-mini contracts per VIX contract (equation 4), set at entry."""
        mkt = self.mkt
        b = self.coefs.get(mkt.dates[i].year)
        if b is None or i == 0:
            return None
        tts_lag = mkt.tts(i - 1, k)
        vix_usd = (b[1] + b[2] * tts_lag) * mkt.cfg.vix_mult      # per 1% S&P
        es_usd = 0.01 * mkt.es[i - 1] * mkt.cfg.es_mult           # per 1% S&P
        return abs(vix_usd / es_usd)


# --------------------------------------------------------------------------
# Positions and P&L
# --------------------------------------------------------------------------
@dataclass
class Position:
    side: int            # -1 short VIX (+ short ES), +1 long VIX (+ long ES)
    k: pd.Timestamp      # VIX futures contract (expiry)
    i: int               # entry index
    vix_fill: float
    es_fill: float
    hr: float
    entry_roll: float
    last_roll: float
    roll_pnl: float = 0.0


def open_position(mkt: Market, hm: HedgeModel, i: int, side: int) -> Position:
    k = mkt.selected[i]
    b, a = mkt.quote(i, k)
    roll = mkt.daily_roll(i, k)
    return Position(side=side, k=k, i=i,
                    vix_fill=b if side < 0 else a,          # pay full spread
                    es_fill=mkt.es[i], hr=hm.ratio(i, k),
                    entry_roll=roll, last_roll=roll)


def mark(mkt: Market, pos: Position, i: int) -> tuple[float, float]:
    """(VIX-leg P&L, hedge-leg P&L) if closed at i, net of all costs."""
    c = mkt.cfg
    b, a = mkt.quote(i, pos.k)
    exit_fill = a if pos.side < 0 else b
    vix_pnl = pos.side * (exit_fill - pos.vix_fill) * c.vix_mult - c.vix_brokerage_rt
    hedge_pnl = (pos.side * pos.hr * (mkt.es[i] - pos.es_fill) * c.es_mult
                 - pos.hr * (2 * c.es_half_tick_usd + c.es_brokerage_rt))
    return vix_pnl, hedge_pnl


# --------------------------------------------------------------------------
# Entry and exit rules
# --------------------------------------------------------------------------
def entry_signal(mkt: Market, hm: HedgeModel, i: int) -> int:
    """+1 long, -1 short, 0 no trade."""
    c = mkt.cfg
    k = mkt.selected[i]
    if k is None or mkt.quote(i, k) is None or hm.ratio(i, k) is None:
        return 0
    state, _, roll = mkt.term_structure_state(i, k)
    if state == "CONTANGO" and roll > c.entry_threshold:
        side, median = -1, c.short_vix_median
    elif state == "BACKWARDATION" and roll < -c.entry_threshold:
        side, median = +1, c.long_vix_median
    else:
        return 0
    if c.vix_filter == "above" and not mkt.vix[i] > median:
        return 0
    if c.vix_filter == "below" and not mkt.vix[i] < median:
        return 0
    return side


def exit_reason(mkt: Market, pos: Position, i: int, roll: float,
                net_pnl: float) -> str | None:
    c = mkt.cfg
    if i - pos.i >= c.max_hold:
        return "5-day"
    if c.exit_mode == "roll":
        if (pos.side < 0 and roll < c.exit_roll) or (pos.side > 0 and roll > -c.exit_roll):
            return "roll"
    elif c.exit_mode == "stop" and net_pnl < -c.stop_loss:
        return "stop"
    elif c.exit_mode == "take" and net_pnl > c.take_gain:
        return "take"
    return None


# --------------------------------------------------------------------------
# Backtest loop
# --------------------------------------------------------------------------
def run_backtest(mkt: Market, hm: HedgeModel) -> pd.DataFrame:
    trades, pos = [], None
    for i in range(len(mkt.dates)):
        if pos is not None:
            if mkt.quote(i, pos.k) is None:      # no quote today: hold
                continue
            # roll earned over the previous day, recomputed daily (gross)
            pos.roll_pnl += -pos.side * pos.last_roll * mkt.cfg.vix_mult
            roll = mkt.daily_roll(i, pos.k)
            pos.last_roll = roll
            vix_pnl, hedge_pnl = mark(mkt, pos, i)
            reason = exit_reason(mkt, pos, i, roll, vix_pnl + hedge_pnl)
            if reason:
                trades.append(dict(
                    entry_date=mkt.dates[pos.i], exit_date=mkt.dates[i],
                    side="SHORT" if pos.side < 0 else "LONG",
                    contract=pos.k.date(), days=i - pos.i,
                    vix_at_entry=mkt.vix[pos.i], entry_roll=pos.entry_roll,
                    hedge_ratio=pos.hr, exit_reason=reason,
                    hedged=vix_pnl + hedge_pnl, unhedged=vix_pnl,
                    hedge=hedge_pnl, roll=pos.roll_pnl))
                pos = None
            continue                              # no same-day re-entry
        side = entry_signal(mkt, hm, i)
        if side:
            pos = open_position(mkt, hm, i, side)
    if pos is not None:
        print(f"  [note] {'SHORT' if pos.side < 0 else 'LONG'} position opened "
              f"{mkt.dates[pos.i].date()} is still open at the end of the data - not counted.")
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------
# Statistics and bootstrap
# --------------------------------------------------------------------------
def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    semi = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))   # gains set to zero
    return {"Mean": x.mean(), "Semi Std Dev": semi,
            "90% fractile": np.percentile(x, 90),
            "10% fractile": np.percentile(x, 10),
            "Winners/Losers": f"{(x > 0).sum()}/{(x <= 0).sum()}",
            "Sortino": x.mean() / semi if semi > 0 else np.nan}


class RandomEntryPool:
    """P&L of hedged trades opened on every eligible day, held a fixed time."""

    def __init__(self, mkt: Market, hm: HedgeModel):
        self.mkt, self.hm, self.cache = mkt, hm, {}

    def get(self, side: int, hold: int) -> np.ndarray:
        key = (side, hold)
        if key not in self.cache:
            mkt, out = self.mkt, []
            for i in range(1, len(mkt.dates) - hold):
                k = mkt.selected[i]
                if k is None or mkt.quote(i, k) is None or self.hm.ratio(i, k) is None:
                    continue
                if mkt.quote(i + hold, k) is None:
                    continue
                pos = open_position(mkt, self.hm, i, side)
                v, h = mark(mkt, pos, i + hold)
                out.append((v + h, v, h))
            self.cache[key] = np.array(out)
        return self.cache[key]


def bootstrap_p(pool: np.ndarray, actual_mean: float, n: int,
                draws: int, rng: np.random.Generator) -> float:
    """Share of random-entry samples whose mean beats the actual mean."""
    if len(pool) == 0 or n == 0:
        return np.nan
    means = np.empty(draws)
    batch = max(1, 2_000_000 // n)
    for s in range(0, draws, batch):
        m = min(batch, draws - s)
        means[s:s + m] = pool[rng.integers(0, len(pool), size=(m, n))].mean(axis=1)
    return float((means >= actual_mean).mean())


def report(trades: pd.DataFrame, pool: RandomEntryPool, cfg: Config) -> None:
    rng = np.random.default_rng(cfg.seed)
    for side_name, side in (("SHORT", -1), ("LONG", +1)):
        t = trades[trades["side"] == side_name] if len(trades) else trades
        print(f"\n  {side_name} trades: {len(t)}")
        if len(t) == 0:
            continue
        hold = max(1, int(round(t["days"].mean())))
        rand = pool.get(side, hold)
        cols = {}
        for j, (label, col) in enumerate((("Hedged P&L", "hedged"),
                                          ("Unhedged P&L", "unhedged"),
                                          ("S&P Hedge P&L", "hedge"))):
            d = describe(t[col].to_numpy())
            col_pool = rand[:, j] if rand.ndim == 2 else np.empty(0)
            d["P-value"] = bootstrap_p(col_pool, d["Mean"], len(t),
                                       cfg.bootstrap_draws, rng)
            cols[label] = d
        r = describe(t["roll"].to_numpy())
        cols["Roll P&L (gross)"] = {"Mean": r["Mean"], "90% fractile": r["90% fractile"],
                                    "10% fractile": r["10% fractile"]}
        order = ["Mean", "P-value", "Semi Std Dev", "90% fractile", "10% fractile",
                 "Winners/Losers", "Sortino"]
        table = pd.DataFrame(cols).reindex(order)
        print(table.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="--"))
        corr = np.corrcoef(t["unhedged"], t["hedge"])[0, 1] if len(t) > 2 else np.nan
        early = (t["exit_reason"] != "5-day").sum()
        print(f"  Avg days in trade: {t['days'].mean():.1f}  |  early exits: {early}"
              f"  |  avg hedge ratio: {t['hedge_ratio'].mean():.2f}"
              f"  |  corr(VIX leg, hedge leg): {corr:.2f}"
              f"  |  bootstrap hold: {hold}d")


# --------------------------------------------------------------------------
# Synthetic data (for testing the pipeline only - results are meaningless)
# --------------------------------------------------------------------------
def make_synthetic(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2006-01-02", "2011-12-30")
    n = len(dates)
    theta_p, theta_q, kappa, volvol, rho = 19.0, 22.0, 0.025, 0.065, -0.75
    x, es_lvl = np.log(18.0), 1250.0
    vix, es = np.empty(n), np.empty(n)
    for t in range(n):
        if t > 0:
            z1, z2 = rng.standard_normal(2)
            jump = rng.random() < 0.004
            ret = 0.0002 + vix[t - 1] / 100 / np.sqrt(252) * z1
            if jump:
                ret -= abs(rng.normal(0.04, 0.02))
            es_lvl *= np.exp(ret)
            x += (kappa * (np.log(theta_p) - x)
                  + volvol * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
                  + (0.35 if jump else 0.0))
        vix[t], es[t] = np.exp(x), es_lvl
    spot = pd.DataFrame({"date": dates, "vix": vix.round(2), "es": (es * 4).round() / 4})

    months = pd.date_range("2005-12-01", "2012-10-01", freq="MS")
    expiries = [m + pd.offsets.WeekOfMonth(week=2, weekday=2) for m in months]
    d64 = dates.values.astype("datetime64[D]")
    rows = []
    for i, d in enumerate(dates):
        live = [e for e in expiries if e > d][:8]
        for e in live:
            tau = np.busday_count(d64[i], np.datetime64(e, "D"))
            f = theta_q + (vix[i] - theta_q) * np.exp(-kappa * tau)
            f += rng.normal(0, 0.10)
            spread = 0.10 if rng.random() < 0.1 else 0.05
            bid = max(np.floor(f / 0.05) * 0.05, 0.05)
            rows.append((d, e, round(bid, 2), round(bid + spread, 2)))
    fut = pd.DataFrame(rows, columns=["date", "expiry", "bid", "ask"])
    return spot, fut


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Simon & Campasano (2014) VIX basis backtest")
    ap.add_argument("--spot", help="CSV with date, vix, es")
    ap.add_argument("--futures", help="CSV with date, expiry, bid, ask")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--exit-mode", default="all", choices=["time", "roll", "stop", "take", "all"])
    ap.add_argument("--vix-filter", default="none", choices=["none", "above", "below"])
    ap.add_argument("--entry-threshold", type=float, default=0.10)
    ap.add_argument("--draws", type=int, default=10_000, help="bootstrap draws")
    ap.add_argument("--trades-out", help="write trades of the last run to CSV")
    ap.add_argument("--plot", help="save cumulative hedged P&L chart (PNG) of the last run")
    args = ap.parse_args()

    if args.demo:
        spot, fut = make_synthetic()
        print("Using SYNTHETIC data - this only tests the code; numbers mean nothing.")
    elif args.spot and args.futures:
        spot, fut = pd.read_csv(args.spot), pd.read_csv(args.futures)
    else:
        ap.error("give --spot and --futures, or --demo")

    base = Config(entry_threshold=args.entry_threshold, vix_filter=args.vix_filter,
                  bootstrap_draws=args.draws)
    mkt = Market(spot, fut, base)
    hm = HedgeModel(mkt)
    pool = RandomEntryPool(mkt, hm)

    print(f"\nSample: {mkt.dates[0].date()} to {mkt.dates[-1].date()} "
          f"({len(mkt.dates)} days)")
    print("\nHedge model  dF = b0 + b1*SPRET + b2*SPRET*TTS  (expanding window, out of sample)")
    print(pd.DataFrame(hm.fit_info).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    modes = ["time", "roll", "stop", "take"] if args.exit_mode == "all" else [args.exit_mode]
    labels = {"time": "5-day exits (baseline)",
              "roll": "Exit if roll decays past +/-0.05, else 5 days",
              "stop": "Exit if net hedged loss > $500, else 5 days",
              "take": "Exit if net hedged gain > $500, else 5 days"}
    trades = pd.DataFrame()
    for mode in modes:
        cfg = replace(base, exit_mode=mode)
        mkt.cfg = cfg
        trades = run_backtest(mkt, hm)
        print("\n" + "=" * 78)
        print(f"{labels[mode]}  |  entry |roll| > {cfg.entry_threshold:.2f}"
              f"  |  VIX filter: {cfg.vix_filter}")
        print("=" * 78)
        if trades.empty:
            print("  No trades.")
            continue
        report(trades, pool, cfg)

    if args.trades_out and not trades.empty:
        trades.to_csv(args.trades_out, index=False)
        print(f"\nTrades written to {args.trades_out}")
    if args.plot and not trades.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = trades.sort_values("exit_date")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(s["exit_date"], s["hedged"].cumsum(), label="Hedged")
        ax.plot(s["exit_date"], s["unhedged"].cumsum(), label="Unhedged (VIX leg)", alpha=0.7)
        ax.set_ylabel("Cumulative P&L ($, 1 VIX contract)")
        ax.set_title("Hedged VIX basis strategy - cumulative P&L")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Chart saved to {args.plot}")


if __name__ == "__main__":
    main()
