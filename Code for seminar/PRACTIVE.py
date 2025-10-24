# -*- coding: utf-8 -*-
# Regimes: threshold trigger (contradiction-only) + other-leg L-day avg (causal)
# Portfolio universe: S&P 500 (price), 10Y bond (duration approx), Gold (price)
# Train/Test split: TRAIN ends at TRAIN_END_DATE (inclusive); TEST starts the next trading day (true OOS)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= User settings =========
# File locations and column names you have in Excel. DATE must be parseable.
FILE = "/Users/asgerboerasmussen/Desktop/Seminar/Samlet data, daily.xlsx"
SHEET = "Ark1"
DATE_COL = "DATE"

# Limit the analysis to a window (useful to cut off very old data or incomplete recent data).
START_DATE = "2000-01-01"
END_DATE   = "2025-01-01"

# TRAIN/TEST split: give a date string for the last day in TRAIN. TEST starts the day after.
# If you set this to None, the code will auto-pick the midpoint after data is prepared.
TRAIN_END_DATE_STR = "2020-06-30"

# --- Regime engine parameters ---
# SPAN: smoother for EWMA. Larger => smoother series => fewer, slower regime changes.
# THRESH_BPS: width of the trigger band in basis points (contradiction-only).
# L: how many days we average the *direction* on the other leg to determine the new regime.
SPAN        = 100
THRESH_BPS  = 15
L           = 10

# Transaction costs: proportional to turnover, one-way.
# Example: 50 bps = 0.50% *per side* you trade on the *marginal* rebalance amount.
TC_BPS  = 50.0
TC_RATE = TC_BPS / 10000.0

# ========= Load & pre-clean =========
# Read Excel, sort by date, set the index to date so time slicing is easy.
df = pd.read_excel(FILE, sheet_name=SHEET, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).set_index(DATE_COL)

# Limit the data to the study window (after loading). This way all derived series use only this window.
if START_DATE or END_DATE:
    start = pd.to_datetime(START_DATE) if START_DATE else df.index.min()
    end   = pd.to_datetime(END_DATE)   if END_DATE   else df.index.max()
    df = df.loc[start:end]

# We use the 10Y/2Y curve. If missing, compute:
#   DifY = 10Y - 2Y (yield curve slope)
#   AvgY = (10Y + 2Y) / 2 (average level of the two points)
if "DifY" not in df: df["DifY"] = df["10Y"] - df["2Y"]
if "AvgY" not in df: df["AvgY"] = (df["10Y"] + df["2Y"]) / 2.0

# The trigger band is given in bps but your yields might be in %
# If AvgY median < 1, likely in decimal (e.g., 0.0423). Then 1 unit = 1.0 => 10,000 bps.
# Else yields are in percent, so 1 unit = 1% => 100 bps.
BPS_PER_UNIT = 10000 if df["AvgY"].dropna().median() < 1.0 else 100
thr = THRESH_BPS / BPS_PER_UNIT  # threshold in the *units* of AvgY/DifY

# --- Smooth series (EWMA) ---
# We smooth the *inputs to the regime engine* to avoid noise.
# ewm(span=SPAN) is a classic exponentially weighted moving average.
for col in ["10Y", "2Y", "AvgY", "DifY"]:
    df[f"{col}_smooth"] = df[col].ewm(span=SPAN, adjust=False).mean()

# Pull numpy arrays for speed inside the regime loop.
A = df["AvgY_smooth"].to_numpy(dtype=float)   # level leg
S = df["DifY_smooth"].to_numpy(dtype=float)   # slope leg

# Direction cues: we don’t flip regime on noise.
# We look at the *L-day average of differences* to infer a more persistent direction.
# A_trend[t] = mean over last L days of ΔA; same for S.
A_trend = df["AvgY_smooth"].diff().rolling(L).mean().to_numpy(dtype=float)
S_trend = df["DifY_smooth"].diff().rolling(L).mean().to_numpy(dtype=float)

def nz_sign(x, fallback):
    """
    Robust sign: if x is 0/NaN, use the fallback sign instead.
    Why: early in the sample or in flat periods, the trend can be 0/NaN.
    """
    if np.isnan(x) or x == 0:
        return 1 if fallback > 0 else -1 if fallback < 0 else 1
    return 1 if x > 0 else -1

def quad_from_signs(sa, ss):
    """
    Map (sign of AvgY, sign of DifY) to the 4 canonical yield-curve regimes:
      +/+  = Bear Steepening      (rates up, slope up)
      +/−  = Bear Flattening      (rates up, slope down)
      −/+  = Bull Steepening      (rates down, slope up)
      −/−  = Bull Flattening      (rates down, slope down)
    """
    if sa > 0 and ss > 0:  return "Bear Steepening"
    if sa > 0 and ss < 0:  return "Bear Flattening"
    if sa < 0 and ss > 0:  return "Bull Steepening"
    if sa < 0 and ss < 0:  return "Bull Flattening"
    return None

# For the *current* regime we also know what we "expect" each sign to be.
sign_map = {
    "Bear Steepening": ( 1,  1),
    "Bear Flattening": ( 1, -1),
    "Bull Steepening": (-1,  1),
    "Bull Flattening": (-1, -1),
}

# --- Regime engine ---
# Key idea: A regime switch only happens when the *trigger leg* (either AvgY or DifY)
# moves by ≥ threshold (thr) *in the opposite direction* of the current regime ("contradiction").
# The *other* leg’s direction is inferred from the L-day average to classify the new quadrant.
n = len(df)
R = np.empty(n, dtype=object)

# The "anchor" is the last reference value for each leg; we measure moves relative to it.
# On same-direction pushes, we *re-anchor* (to avoid multiple triggers from a single drift).
i0 = int(np.argmax(~(np.isnan(A) | np.isnan(S))))  # first valid index
anchorA, anchorS = A[i0], S[i0]

# Initial regime from early trends (fallback to immediate deltas if needed)
sa0 = nz_sign(A_trend[i0], A[i0] - anchorA)
ss0 = nz_sign(S_trend[i0], S[i0] - anchorS)
curr = quad_from_signs(sa0, ss0)
R[:i0+1] = curr

for t in range(i0+1, n):
    if np.isnan(A[t]) or np.isnan(S[t]):
        R[t] = curr; continue

    # Distance from anchors (how far has each leg moved since its anchor?)
    dA = A[t] - anchorA
    dS = S[t] - anchorS

    # Expected signs under the current regime (if unknown, fall back to initial signs)
    expA, expS = sign_map.get(curr, (sa0, ss0))

    # If A-leg moved more than thr:
    if abs(dA) >= thr:
        sa = 1 if dA > 0 else -1
        if curr is None or sa != expA:
            # CONTRADICTION → evaluate switch; classify with other leg's L-day average
            ss = nz_sign(S_trend[t], S[t] - anchorS)
            cand = quad_from_signs(sa, ss)
            if cand and cand != curr:
                curr = cand
                anchorA, anchorS = A[t], S[t]  # reset anchors at switch
        else:
            # Same-direction push → just move the A anchor forward (prevents multi-triggers)
            anchorA = A[t]

    # Else if S-leg moved more than thr:
    elif abs(dS) >= thr:
        ss = 1 if dS > 0 else -1
        if curr is None or ss != expS:
            sa = nz_sign(A_trend[t], A[t] - anchorA)
            cand = quad_from_signs(sa, ss)
            if cand and cand != curr:
                curr = cand
                anchorA, anchorS = A[t], S[t]
        else:
            # Same-direction push → move the S anchor
            anchorS = S[t]

    R[t] = curr

df["Regime"] = pd.Series(R, index=df.index)
print(df["Regime"].value_counts(dropna=False))

# ========= Return construction =========
# Price assets (S&P500, Gold): simple close-to-close percent returns.
df["ret_spx"]  = df["S&P500"].pct_change()
df["ret_gold"] = df["Gold"].pct_change()

# Bond returns using a duration-based linear approximation around small yield moves:
#   ΔPrice ≈ -Duration * ΔYield  (plus daily carry)
# -Duration: price falls when yields rise.
# carry_t: yesterday's yield / 252 (simple approximation to daily accrual)
y2  = (df["2Y"]  / 100.0).astype(float)
y10 = (df["10Y"] / 100.0).astype(float)
dy2, dy10 = y2.diff(), y10.diff()

DUR_2Y, DUR_10Y = 1.9, 8.8
USE_CARRY = True
carry2  = (y2.shift(1)  / 252.0) if USE_CARRY else 0.0
carry10 = (y10.shift(1) / 252.0) if USE_CARRY else 0.0

df["ret_2Y"]  = (-DUR_2Y  * dy2)  + carry2
df["ret_10Y"] = (-DUR_10Y * dy10) + carry10

# Align returns to the regime that was *active yesterday*.
# This avoids taking returns on the *switch day* with look-ahead.
df["Regime_trade"] = df["Regime"].shift(1)

# Keep only days where all needed columns exist (and Regime_trade exists).
trade = df.dropna(subset=["Regime_trade","ret_spx","ret_10Y","ret_gold"]).copy()

# ========= Train / Test split =========
# We parse TRAIN_END_DATE_STR (if provided) into a Timestamp and clamp to the observed range.
# If None or invalid, we choose the midpoint of the *observed* trade index so both halves are balanced.
if TRAIN_END_DATE_STR:
    TRAIN_END_DATE = pd.to_datetime(TRAIN_END_DATE_STR)
    if TRAIN_END_DATE < trade.index.min() or TRAIN_END_DATE > trade.index.max():
        TRAIN_END_DATE = trade.index[len(trade)//2 - 1]
else:
    TRAIN_END_DATE = trade.index[len(trade)//2 - 1]

trade_train = trade.loc[:TRAIN_END_DATE]                # TRAIN includes the end date
trade_test  = trade.loc[trade.index > TRAIN_END_DATE]   # TEST starts the next day

print(f"\nTRAIN_END_DATE: {TRAIN_END_DATE.date()}")
print(f"Train days: {len(trade_train):,}  |  Test days (OOS): {len(trade_test):,}")

# ========= Plot: regimes over time (helps to see switches and split line) =========
colors = {
    "Bull Steepening": "#1f77b4",
    "Bull Flattening": "#2ca02c",
    "Bear Steepening": "#d62728",
    "Bear Flattening": "#ffe761",
}
plt.figure(figsize=(12,5))
plt.plot(df.index, df["DifY_smooth"], linewidth=1.8, label="Smoothed Yield Curve (10Y−2Y)")
plt.plot(df.index, df["AvgY_smooth"], linewidth=1.8, label="Smoothed Average Yield")
ymin = np.nanmin(np.vstack([df["DifY_smooth"].values, df["AvgY_smooth"].values]))
ymax = np.nanmax(np.vstack([df["DifY_smooth"].values, df["AvgY_smooth"].values]))
added = set()
for regime, color in colors.items():
    mask = df["Regime"].eq(regime)
    lbl = regime if regime not in added else None
    plt.fill_between(df.index, ymin, ymax, where=mask, color=color, alpha=0.12, label=lbl)
    added.add(regime)
plt.axvline(TRAIN_END_DATE, color="k", linestyle="--", alpha=0.6, label="TRAIN_END_DATE")
plt.title(f"Regimes (EWMA span={SPAN}; Trigger ≥ {THRESH_BPS} bps on contradiction; other leg uses {L}-day avg)")
plt.ylabel("Yield"); plt.legend(ncol=2, fontsize=9); plt.tight_layout(); plt.show()

# ========= Plot: $1 growth by asset (full sample) =========
# This shows baseline behavior of the three assets over the whole time (no regime conditioning).
growth_base = df[["ret_spx","ret_10Y","ret_gold"]].dropna().copy()
growth_curves = (1.0 + growth_base).cumprod()
plt.figure(figsize=(10,4))
plt.plot(growth_curves.index, growth_curves["ret_spx"],  label="S&P 500", linewidth=1.6)
plt.plot(growth_curves.index, growth_curves["ret_10Y"],  label="10Y bond (dur approx)", linewidth=1.6)
plt.plot(growth_curves.index, growth_curves["ret_gold"], label="Gold", linewidth=1.6)
plt.title("$1 growth by asset — full sample (gross)")
plt.ylabel("Value of $1"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.show()

# ========= Helpers =========
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

def ann_stats_from_daily(r: pd.Series):
    """
    Convert daily series r into annualized stats:
      E[r_annual] = 252 * mean(r_d)
      σ_annual     = sqrt(252) * std(r_d)
      Sharpe       = E[r_annual] / σ_annual
    No risk-free subtraction here (implicitly 0 RF for comparison).
    """
    mu_d, sd_d = r.mean(), r.std(ddof=1)
    ann_ret = mu_d * 252
    ann_vol = sd_d * np.sqrt(252) if pd.notna(sd_d) else np.nan
    ann_sh  = (ann_ret / ann_vol) if (ann_vol and ann_vol != 0) else np.nan
    return ann_ret, ann_vol, ann_sh

def summarize_daily(g: pd.DataFrame, col: str):
    """Same as above but scoped to a sub-DataFrame g and one return column."""
    mu_d = g[col].mean()
    sd_d = g[col].std(ddof=1)
    ann_ret = mu_d * 252
    ann_vol = sd_d * np.sqrt(252) if pd.notna(sd_d) else np.nan
    sharpe  = (ann_ret / ann_vol) if (ann_vol and ann_vol != 0) else np.nan
    return ann_ret, ann_vol, sharpe

# Sharpe-maximization for 3 assets, long-only, weights sum to 1.
# We solve by a coarse grid search for stability and simplicity.
GRID_STEP   = 0.02
MEAN_SHRINK = 0.50  # shrink means toward 0 to reduce overfitting to TRAIN noise

def sharpe_max_3asset(mu: np.ndarray, cov: np.ndarray, step=GRID_STEP):
    """
    Goal: pick w = (w_spx, w_10y, w_gold) with w>=0, sum=1 to maximize Sharpe = (w'μ)/sqrt(w'Σw).
    Method: grid over (w_spx, w_gold); set w_10y = 1 - w_spx - w_gold. Reject negative w_10y.
    """
    best = (0.0, 0.0, 1.0, -np.inf)  # (w_spx, w_10y, w_gold, sharpe)
    for w_spx in np.arange(0.0, 1.0 + 1e-9, step):
        max_gold = 1.0 - w_spx
        for w_gold in np.arange(0.0, max_gold + 1e-9, step):
            w_10y = 1.0 - w_spx - w_gold
            w = np.array([w_spx, w_10y, w_gold])
            m  = float(w @ mu)
            v  = float(w @ cov @ w)
            if v <= 0: 
                continue
            sh = m / np.sqrt(v)
            if sh > best[3]:
                best = (w_spx, w_10y, w_gold, sh)
    return best

# ========= TRAIN — estimate per-regime weights and a TRAIN-global benchmark =========
REGIMES = ["Bull Steepening","Bull Flattening","Bear Steepening","Bear Flattening"]

weights = {}
rows_w  = []
for reg in REGIMES:
    # Train-only returns for the three assets *conditioned on yesterday's regime*
    sub = trade_train.loc[trade_train["Regime_trade"] == reg, ["ret_spx","ret_10Y","ret_gold"]].dropna()
    if len(sub) == 0:
        # If a regime doesn’t appear in TRAIN, use a neutral fallback (equal weights)
        w_spx, w_10y, w_gold = (1/3, 1/3, 1/3)
    else:
        # Mean shrink: (1-λ)*sample_mean to avoid overly aggressive weights from small samples
        mu  = (1.0 - MEAN_SHRINK) * sub.mean().values
        cov = sub.cov().values
        w_spx, w_10y, w_gold, _ = sharpe_max_3asset(mu, cov)
    weights[reg] = (w_spx, w_10y, w_gold)
    rows_w.append({"regime": reg, "n_train_days": len(sub), "w_spx": w_spx, "w_10y": w_10y, "w_gold": w_gold})

weights_per_regime = pd.DataFrame(rows_w).set_index("regime").reindex(REGIMES)
print("\n=== Per-regime Sharpe-max weights (TRAIN ONLY; SPX/10Y/Gold, long-only) ===")
print(weights_per_regime.to_string())

# Global benchmark: same Sharpe maximization but without conditioning on a regime
sub_all_train = trade_train[["ret_spx","ret_10Y","ret_gold"]].dropna()
mu_all  = (1.0 - MEAN_SHRINK) * sub_all_train.mean().values
cov_all = sub_all_train.cov().values
w_spx_g, w_10y_g, w_gold_g, _ = sharpe_max_3asset(mu_all, cov_all)
print("\n=== Global Sharpe-max weights (TRAIN ONLY) ===")
print(f"w_SPX={w_spx_g:,.3f}  w_10Y={w_10y_g:,.3f}  w_Gold={w_gold_g:,.3f}")

# ========= TRAIN — asset performance by regime (table + bar chart) =========
# This shows how each *single* asset behaved in each regime (TRAIN only).
asset_cols = [("ret_spx","S&P 500"), ("ret_10Y","10Y bond (dur approx)"), ("ret_gold","Gold")]
rows_tr = []
for reg in REGIMES:
    sub = trade_train.loc[trade_train["Regime_trade"] == reg]
    for c, label in asset_cols:
        if len(sub) == 0:
            rows_tr.append({"regime": reg, "asset": label, "n_days": 0,
                            "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan})
        else:
            ar, av, sh = summarize_daily(sub, c)
            rows_tr.append({"regime": reg, "asset": label, "n_days": len(sub),
                            "ann_return": ar, "ann_vol": av, "sharpe": sh})

perf_train = pd.DataFrame(rows_tr).sort_values(["asset","regime"])
print("\n=== TRAIN — Asset performance by regime (daily stats; trade from day after signal) ===")
print(perf_train.to_string(index=False))

pivot_ret_tr = perf_train.pivot(index="regime", columns="asset", values="ann_return").reindex(REGIMES)
ax = pivot_ret_tr.plot(kind="bar", figsize=(10,5))
ax.set_title("TRAIN — Annualized return by regime & asset")
ax.set_ylabel("Annualized return")
ax.legend(title=""); plt.tight_layout(); plt.show()

# ========= TRAIN — weights bar chart (per-regime + global benchmark) =========
weights_plot = (
    weights_per_regime[["w_spx","w_10y","w_gold"]]
      .rename(columns={"w_spx":"S&P 500","w_10y":"10Y bond","w_gold":"Gold"})
      .copy()
)
weights_plot.loc["Global (TRAIN)"] = [w_spx_g, w_10y_g, w_gold_g]

plt.figure(figsize=(10,4))
weights_plot.plot(kind="bar", ax=plt.gca())
plt.title("TRAIN — Sharpe-max weights per regime vs Global benchmark")
plt.ylabel("Weight"); plt.ylim(0,1); plt.legend(title=""); plt.tight_layout(); plt.show()

# ========= OOS BACKTEST ON TEST ONLY =========
# We now *freeze* the per-regime weights found on TRAIN and apply them out of sample.
trade_test = trade_test.copy()
trade_test["w_spx_tgt"]  = trade_test["Regime_trade"].map({k:v[0] for k,v in weights.items()})
trade_test["w_10y_tgt"]  = trade_test["Regime_trade"].map({k:v[1] for k,v in weights.items()})
trade_test["w_gold_tgt"] = trade_test["Regime_trade"].map({k:v[2] for k,v in weights.items()})

# Gross (no-cost) daily portfolio return = dot product of target weights and daily returns
trade_test["ret_pf"] = (
    trade_test["w_spx_tgt"]  * trade_test["ret_spx"] +
    trade_test["w_10y_tgt"]  * trade_test["ret_10Y"] +
    trade_test["w_gold_tgt"] * trade_test["ret_gold"]
)

def apply_tc_daily_N(tr: pd.DataFrame, w_cols, r_cols, tc_rate=TC_RATE):
    """
    Daily rebalancing with linear proportional costs on *turnover*.
    Steps each day t:
      1) Drift yesterday's *post-rebalance* weights with yesterday returns:
         w_pre_i = (w_post_{t-1,i} * (1 + r_{t-1,i})) / sum_j(...)
      2) Compute turnover_t = 0.5 * sum_i |w_target_i - w_pre_i|
         (The 0.5 makes it "one-way notional"; with weights summing to 1, this equals net traded.)
      3) Cost_t = tc_rate * turnover_t    (e.g., 50 bps * turnover)
      4) Net return_t = (w_target · r_t) - Cost_t
    """
    cols = w_cols + r_cols
    tr = tr[cols].dropna().copy()
    idx = tr.index
    k = len(w_cols)

    ret_net, turns, costs = [], [], []

    # Initialize: on the first day we assume we already hold today's target (no cost on day 1).
    last_w_post = tr[w_cols].iloc[0].to_numpy(dtype=float)
    last_r      = np.zeros(k, dtype=float)

    for i in range(len(tr)):
        w_tgt = tr[w_cols].iloc[i].to_numpy(dtype=float)
        r     = tr[r_cols].iloc[i].to_numpy(dtype=float)

        if i == 0:
            w_pre = last_w_post
        else:
            # Drift last day's post weights by last day's returns, then renormalize to sum to 1.
            num = last_w_post * (1.0 + last_r)
            den = num.sum()
            w_pre = num / den if den != 0 else last_w_post

        turnover = 0.5 * np.abs(w_tgt - w_pre).sum()
        cost = tc_rate * turnover

        r_gross = float((w_tgt * r).sum())
        r_net   = r_gross - cost

        ret_net.append(r_net); turns.append(turnover); costs.append(cost)

        # After rebalancing, our new post-weights are exactly the targets we chose.
        last_w_post = w_tgt
        last_r = r

    return (
        pd.Series(ret_net, index=idx, name="ret_net"),
        pd.Series(turns,   index=idx, name="turnover"),
        pd.Series(costs,   index=idx, name="tc_paid"),
    )

# Apply TC engine to TEST (this gives you net returns and turnover diagnostics)
trade_test["ret_pf_net"], trade_test["turnover"], trade_test["tc_paid"] = apply_tc_daily_N(
    trade_test,
    w_cols=["w_spx_tgt","w_10y_tgt","w_gold_tgt"],
    r_cols=["ret_spx","ret_10Y","ret_gold"],
    tc_rate=TC_RATE
)

# Global benchmark out of sample: freeze the TRAIN-global weights and apply on TEST
trade_test["w_spx_tgt_global"]  = w_spx_g
trade_test["w_10y_tgt_global"]  = w_10y_g
trade_test["w_gold_tgt_global"] = w_gold_g
trade_test["ret_pf_global"] = (
    trade_test["w_spx_tgt_global"]  * trade_test["ret_spx"] +
    trade_test["w_10y_tgt_global"]  * trade_test["ret_10Y"] +
    trade_test["w_gold_tgt_global"] * trade_test["ret_gold"]
)
trade_test["ret_pf_global_net"], _, _ = apply_tc_daily_N(
    trade_test,
    w_cols=["w_spx_tgt_global","w_10y_tgt_global","w_gold_tgt_global"],
    r_cols=["ret_spx","ret_10Y","ret_gold"],
    tc_rate=TC_RATE
)

# ========= REPORT: OOS (TEST) ONLY =========
# Summarize the strategy’s out-of-sample performance with and without costs.
ann_ret, ann_vol, ann_sh = ann_stats_from_daily(trade_test["ret_pf"])
ann_ret_n, ann_vol_n, ann_sh_n = ann_stats_from_daily(trade_test["ret_pf_net"])
print(f"\n=== OOS Regime Portfolio (TEST ONLY) — daily rebalance ===")
print(f" Gross  -> AnnRet: {ann_ret:,.3f}  Vol: {ann_vol:,.3f}  Sharpe: {ann_sh:,.3f}")
print(f" Net TC -> AnnRet: {ann_ret_n:,.3f}  Vol: {ann_vol_n:,.3f}  Sharpe: {ann_sh_n:,.3f}   (TC={TC_BPS:.2f} bps one-way)")

# Same for the frozen-global benchmark out of sample.
ann_ret_g, ann_vol_g, ann_sh_g   = ann_stats_from_daily(trade_test["ret_pf_global"])
ann_ret_gn, ann_vol_gn, ann_sh_gn = ann_stats_from_daily(trade_test["ret_pf_global_net"])
print("\n=== OOS Global Sharpe-max (TEST ONLY; weights from TRAIN) ===")
print(f" Gross  -> AnnRet: {ann_ret_g:,.3f}  Vol: {ann_vol_g:,.3f}  Sharpe: {ann_sh_g:,.3f}")
print(f" Net TC -> AnnRet: {ann_ret_gn:,.3f} Vol: {ann_vol_gn:,.3f} Sharpe: {ann_sh_gn:,.3f}   (TC={TC_BPS:.2f} bps)")

# ========= OOS equity curves (TEST only) =========
# These show the effect of costs visually and compare regime vs global.
plt.figure(figsize=(10,4))
eq_regime_g  = (1.0 + trade_test["ret_pf"]).cumprod()
eq_regime_n  = (1.0 + trade_test["ret_pf_net"]).cumprod()
plt.plot(eq_regime_g.index, eq_regime_g.values, linewidth=1.8, label="Regime (Gross)")
plt.plot(eq_regime_n.index, eq_regime_n.values, linewidth=1.5, label=f"Regime (Net, {TC_BPS:.0f} bps)", alpha=0.9)
plt.title("OOS — Regime portfolio (TEST only) — Gross vs Net")
plt.ylabel("Cumulative growth (1→)")
plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
eq_global_g = (1.0 + trade_test["ret_pf_global"]).cumprod()
eq_global_n = (1.0 + trade_test["ret_pf_global_net"]).cumprod()
plt.plot(eq_global_g.index, eq_global_g.values, linewidth=1.8, label="Global (Gross)")
plt.plot(eq_global_n.index, eq_global_n.values, linewidth=1.5, label=f"Global (Net, {TC_BPS:.0f} bps)", alpha=0.9)
plt.title("OOS — Global Sharpe-max (TEST only) — Gross vs Net")
plt.ylabel("Cumulative growth (1→)")
plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(eq_regime_n.index, eq_regime_n.values, linewidth=1.8, label=f"Regime (Net, {TC_BPS:.0f} bps)")
plt.plot(eq_global_n.index, eq_global_n.values, linewidth=1.5, label=f"Global (Net, {TC_BPS:.0f} bps)", alpha=0.9)
plt.title("OOS — Net equity: Regime vs Global (TEST only)")
plt.ylabel("Cumulative growth (1→)")
plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.show()
