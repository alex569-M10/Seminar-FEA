import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from scipy.optimize import minimize
from math import sqrt
from scipy.stats import norm
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

# ========= Setup =========
FILE = "/Users/asgerboerasmussen/Desktop/Seminar/DataDaily.xlsx" #UPDATE TO FIT YOUR PATH
SHEET = "Ark1"
DATE_COL = "DATE"
START_DATE = "1988-01-01"
END_DATE   = "2025-09-01"
TRAIN_END_DATE_STR = "2005-01-01"

# Regime engine params
SPAN = 500       # EWMA length for smoothing AvgY and DifY
threshold = 0.1  # contradiction trigger for re-anchoring

# MVO & bounds
RISK_AVERSION = 4
MIN_W = np.array([ 0.30, 0.35, 0.01, 0.01 ]) # SPX/10Y/Gold/Oil
MAX_W = np.array([ 0.70, 0.55, 0.15, 0.10 ]) # SPX/10Y/Gold/Oil

# Rebalance cadence & execution 
REBAL_FREQ    = "W-FRI"   # Weekly rebalance on Fridays
GAMMA_STICK   = 0.005     # L2 penalty toward previous node's weights

# Naive benchmark
W_NAIVE = np.array([0.50, 0.45, 0.03, 0.02])  # SPX/10Y/Gold/Oil

# Load data
df = pd.read_excel(FILE, sheet_name=SHEET, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).set_index(DATE_COL)
if START_DATE or END_DATE:
    start = pd.to_datetime(START_DATE) if START_DATE else df.index.min()
    end   = pd.to_datetime(END_DATE)   if END_DATE   else df.index.max()
    df = df.loc[start:end]

# ======== Yield curve features for regime engine (10Y–3M slope) ========
if "DifY" not in df:  df["DifY"] = df["10Y"] - df["3M"]           # steepness: 10Y–3M
if "AvgY" not in df:  df["AvgY"] = (df["10Y"] + df["3M"]) / 2.0   # level (average)

# Smooth series for regime classification
for col in ["10Y","3M","AvgY","DifY"]:
    df[f"{col}_smooth"] = df[col].ewm(span=SPAN, adjust=False).mean()

A = df["AvgY_smooth"].to_numpy(float)     # level (rising/falling)
S = df["DifY_smooth"].to_numpy(float)     # slope (steepening/flattening)

def sign_strict(x, default=1):
    if np.isnan(x) or x == 0:
        return default
    return 1 if x > 0 else -1

def label_from_signs(sa, ss):
    # sa: sign of AvgY change → Rising/Falling
    # ss: sign of DifY change → Steepening/Flattening
    if sa > 0 and ss > 0: return "Rising–Steepening"
    if sa > 0 and ss < 0: return "Rising–Flattening"
    if sa < 0 and ss > 0: return "Falling–Steepening"
    if sa < 0 and ss < 0: return "Falling–Flattening"
    return None

sign_map = {
    "Rising–Steepening":   ( 1,  1),
    "Rising–Flattening":   ( 1, -1),
    "Falling–Steepening":  (-1,  1),
    "Falling–Flattening":  (-1, -1),
}

# ============ Regime engine ==========
n = len(df)
R = np.empty(n, dtype=object)
i0 = int(np.argmax(~(np.isnan(A) | np.isnan(S))))
anchorA, anchorS = A[i0], S[i0]

# Initialize regime from first observable move after i0
curr = None
for t in range(i0+1, n):
    if np.isnan(A[t]) or np.isnan(S[t]): 
        continue
    dA0 = A[t] - anchorA
    dS0 = S[t] - anchorS
    if dA0 != 0 or dS0 != 0:
        sa0 = sign_strict(dA0, default=1)
        ss0 = sign_strict(dS0, default=1)
        curr = label_from_signs(sa0, ss0)
        break
if curr is None:
    curr = "Rising–Steepening" 
R[:t+1] = curr

for u in range(t+1, n):
    if np.isnan(A[u]) or np.isnan(S[u]): 
        R[u] = curr
        continue
    dA, dS = A[u]-anchorA, S[u]-anchorS
    expA, expS = sign_map.get(curr, (1, 1))  # expected directions under current regime

    if abs(dA) >= threshold:
        sa = 1 if dA > 0 else -1
        if sa != expA:
            ss = 1 if (S[u]-anchorS) > 0 else -1
            cand = label_from_signs(sa, ss)
            if cand and cand != curr:
                curr = cand
                anchorA, anchorS = A[u], S[u]
        else:
            anchorA = A[u]
    elif abs(dS) >= threshold:
        ss = 1 if dS > 0 else -1
        if ss != expS:
            sa = 1 if (A[u]-anchorA) > 0 else -1
            cand = label_from_signs(sa, ss)
            if cand and cand != curr:
                curr = cand
                anchorA, anchorS = A[u], S[u]
        else:
            anchorS = S[u]
    R[u] = curr

df["Regime"] = pd.Series(R, index=df.index)
df["Regime_trade"] = df["Regime"].shift(1)   # Use yesterday's regime when trading today

# ========= Asset returns =========
df["ret_spx"]  = df["S&P500TR"].pct_change()
df["ret_gold"] = df["Gold"].pct_change()
df["ret_oil"]  = df["Oil"].pct_change()

# 10Y bond return proxy: 
y10 = (df["10Y"].astype(float) / 100.0)
dy10 = y10.diff()
M = 10.0  
y_safe = y10.clip(lower=1e-6)
# Swinkels Eq. (1):
D10 = (1.0 / y_safe) * (1.0 - 1.0 / np.power(1.0 + 0.5 * y_safe, 2.0 * M))
# Swinkels Eq. (2):
C10 = (2.0 / (y_safe ** 2)) * (1.0 - 1.0 / np.power(1.0 + 0.5 * y_safe, 2.0 * M)) - (2.0 * M) / (y_safe * np.power(1.0 + 0.5 * y_safe, 2.0 * M + 1.0))
D_lag = D10.shift(1)
C_lag = C10.shift(1)
carry10 = y10.shift(1) / 252.0
# Swinkels Eq. (3):
df["ret_10Y"] = carry10 - D_lag * dy10 +  0.5 * C_lag * (dy10 ** 2)

# Risk-free daily rate
rf_yield = (df["3M"]/100.0).astype(float)
df["rf_daily"] = rf_yield.shift(1) / 252.0

# ========= FIGURE: Yield curve – 3M, 10Y and slope =========
fig, ax = plt.subplots(figsize=(12, 4.5))
l1, = ax.plot(df.index, df["3M"],  lw=2.0, label="3M yield (%)")
l2, = ax.plot(df.index, df["10Y"], lw=2.0, label="10Y yield (%)")
l3, = ax.plot(df.index, df["10Y"] - df["3M"], lw=1.6, ls=":", color="k", label="10Y - 3M (pp)")
ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_title("Yield curve: short rate, long rate and slope")
ax.set_ylabel("Percent / percentage points")
ax.grid(alpha=0.25)
ax.legend(loc="upper right")
fig.tight_layout(); plt.show()

# ========= Keep only rows with needed cols =========
need_cols = ["Regime_trade","ret_spx","ret_10Y","ret_gold","ret_oil","rf_daily",
             "AvgY_smooth","DifY_smooth"]
trade = df.dropna(subset=need_cols).copy()

# ========= Build LOG EXCESS returns  =========
trade["gross_spx"]  = 1.0 + trade["ret_spx"]
trade["gross_10Y"]  = 1.0 + trade["ret_10Y"]
trade["gross_gold"] = 1.0 + trade["ret_gold"]
trade["gross_oil"]  = 1.0 + trade["ret_oil"]
trade["gross_rf"]   = 1.0 + trade["rf_daily"]

trade["log_spx"]  = np.log(trade["gross_spx"])
trade["log_10Y"]  = np.log(trade["gross_10Y"])
trade["log_gold"] = np.log(trade["gross_gold"])
trade["log_oil"]  = np.log(trade["gross_oil"])
trade["log_rf"]   = np.log(trade["gross_rf"])

trade["lexcess_spx"]  = trade["log_spx"]  - trade["log_rf"]
trade["lexcess_10Y"]  = trade["log_10Y"]  - trade["log_rf"]
trade["lexcess_gold"] = trade["log_gold"] - trade["log_rf"]
trade["lexcess_oil"]  = trade["log_oil"]  - trade["log_rf"]
asset_lexcess_cols = ["lexcess_spx","lexcess_10Y","lexcess_gold","lexcess_oil"]

# ========= Train / Test split =========
if TRAIN_END_DATE_STR:
    TRAIN_END_DATE = pd.to_datetime(TRAIN_END_DATE_STR)
    if (TRAIN_END_DATE < trade.index.min()) or (TRAIN_END_DATE > trade.index.max()):
        TRAIN_END_DATE = trade.index[len(trade)//2 - 1]
else:
    TRAIN_END_DATE = trade.index[len(trade)//2 - 1]

trade_train = trade.loc[:TRAIN_END_DATE].copy()
trade_test  = trade.loc[trade.index > TRAIN_END_DATE].copy()

# ========= FIGURE: Compounded growth of $1 (TEST period) =========
def cum_index_from_returns(r: pd.Series, base: float = 1.0) -> pd.Series:
    """Compound simple daily returns into an index, starting at `base`."""
    return base * (1.0 + r.fillna(0.0)).cumprod()

idx_spx_test  = cum_index_from_returns(trade_test["ret_spx"],  base=1.0)
idx_rf_test   = cum_index_from_returns(trade_test["rf_daily"], base=1.0)   # 3M risk-free
idx_10y_test  = cum_index_from_returns(trade_test["ret_10Y"], base=1.0)
idx_gold_test = cum_index_from_returns(trade_test["ret_gold"], base=1.0)
idx_oil_test  = cum_index_from_returns(trade_test["ret_oil"],  base=1.0)

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(idx_spx_test.index,  idx_spx_test.values,  lw=1.8, label="S&P 500")
ax.plot(idx_rf_test.index,   idx_rf_test.values,   lw=1.6, label="3M bond (risk-free)")
ax.plot(idx_10y_test.index,  idx_10y_test.values,  lw=1.6, label="10Y bond")
ax.plot(idx_gold_test.index, idx_gold_test.values, lw=1.6, label="Gold")
ax.plot(idx_oil_test.index,  idx_oil_test.values,  lw=1.6, label="Oil")
ax.set_title("Compounded growth of $1 — TEST period")
ax.set_ylabel("Value of $1 (rebased at test start)")
ax.grid(alpha=0.25)
ax.legend(loc="upper left")
fig.tight_layout(); plt.show()

# ========= TRAIN stats =========
sub_all_train_ex = trade_train[asset_lexcess_cols].dropna()
Sigma_train = sub_all_train_ex.cov().values     # NO RIDGE
mu_train    = sub_all_train_ex.mean().values

REGIMES = ["Rising–Steepening","Rising–Flattening","Falling–Steepening","Falling–Flattening"]

# ====== MVO with stickiness ======
def feasible_start(min_w, max_w):
    w0 = min_w.astype(float).copy()
    leftover = 1.0 - w0.sum()
    cap = (max_w - min_w)
    if leftover > 0 and cap.sum() > 0: w0 += leftover * (cap / cap.sum())
    w0 = np.clip(w0, min_w, max_w)
    return w0 / w0.sum()

def mv_optimize_box(mu, Sigma, min_w, max_w, risk_aversion=RISK_AVERSION,
                    anchor=None, gamma=0.0):
    """
    Maximize: mu'w - 0.5*lambda*w'Sigma w - gamma*||w - anchor||^2
    s.t. 1'w = 1,  min_w <= w <= max_w
    """
    k = len(mu)
    if anchor is None:
        anchor = 0.5 * (min_w + max_w)
    anchor = np.asarray(anchor, float)

    def obj(w):
        util  = float(w @ mu) - 0.5 * risk_aversion * float(w @ Sigma @ w)
        stick = gamma * float(np.sum((w - anchor)**2))
        return -(util - stick)

    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bnds = tuple((min_w[i], max_w[i]) for i in range(k))
    x0   = feasible_start(min_w, max_w)
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons,
                   options={'maxiter':1000, 'ftol':1e-9, 'disp':False})
    if (not res.success) or np.any(~np.isfinite(res.x)): return np.array(x0)
    return np.array(res.x)

# PRINT: TRAIN-only per-regime weights 
rows_train = []
for reg in REGIMES:
    sample = trade_train.loc[trade_train["Regime_trade"] == reg, asset_lexcess_cols].dropna()
    if len(sample) >= 2:
        mu_reg  = sample.mean().values
        Sig_reg = sample.cov().values
    else:
        mu_reg, Sig_reg = mu_train.copy(), Sigma_train.copy()
    w_train_reg = mv_optimize_box(mu_reg, Sig_reg, MIN_W, MAX_W,
                                  risk_aversion=RISK_AVERSION,
                                  anchor=None, gamma=0.0)
    rows_train.append({"Regime": reg, "Obs": len(sample),
                       "w_SPX": w_train_reg[0], "w_10Y": w_train_reg[1],
                       "w_Gold": w_train_reg[2], "w_Oil": w_train_reg[3]})
train_weights_tbl = pd.DataFrame(rows_train).set_index("Regime")
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\nTRAIN weights per regime (no ridge, no stickiness in print):")
print(train_weights_tbl.to_string())

# PRINT: TRAIN GLOBAL weight
w_train_global = mv_optimize_box(mu_train, Sigma_train, MIN_W, MAX_W,
                                 risk_aversion=RISK_AVERSION, anchor=None, gamma=0.0)
print("\nTRAIN GLOBAL (no-regime) weight vector:")
print(f"SPX={w_train_global[0]:.3f}  10Y={w_train_global[1]:.3f}  Gold={w_train_global[2]:.3f}  Oil={w_train_global[3]:.3f}")

# ========= Rebalance nodes =========
rebal_nodes = trade_test.groupby(pd.Grouper(freq=REBAL_FREQ)).tail(1).index.unique().sort_values()
first_test_day = trade_test.index[0]
if (len(rebal_nodes) == 0) or (rebal_nodes[0] > first_test_day):
    rebal_nodes = rebal_nodes.insert(0, first_test_day)
rebal_nodes = pd.DatetimeIndex(sorted(pd.unique(rebal_nodes)))

# ========= REGIME MODEL =========
cache_by_node = {}
prev_node_weights = {r: None for r in REGIMES}  # stickiness anchor per regime

for t in rebal_nodes:
    hist_all = trade.loc[:t].iloc[:-1]   # ALL history ≤ t−1
    cache_for_t = {}
    for reg in REGIMES:
        sample = hist_all.loc[hist_all["Regime_trade"] == reg, asset_lexcess_cols].dropna()
        if len(sample) >= 2:
            mu_reg  = sample.mean().values
            Sig_reg = sample.cov().values
            anchor  = prev_node_weights[reg] if prev_node_weights[reg] is not None else 0.5*(MIN_W+MAX_W)
            w_reg = mv_optimize_box(mu_reg, Sig_reg, MIN_W, MAX_W,
                                    risk_aversion=RISK_AVERSION,
                                    anchor=anchor, gamma=GAMMA_STICK)
            prev_node_weights[reg] = w_reg.copy()
        else:
            if prev_node_weights[reg] is not None:
                w_reg = prev_node_weights[reg].copy()
            else:
                w_reg = mv_optimize_box(mu_train, Sigma_train, MIN_W, MAX_W,
                                        risk_aversion=RISK_AVERSION)
                prev_node_weights[reg] = w_reg.copy()
        cache_for_t[reg] = w_reg
    cache_by_node[t] = cache_for_t

# Map each TEST day to latest node and select regime weight
nodes_series = pd.Series(rebal_nodes, index=rebal_nodes)
node_index = nodes_series.reindex(trade_test.index).ffill()
if node_index.isna().any(): node_index = node_index.fillna(rebal_nodes[0])

w_tgt = pd.DataFrame(index=trade_test.index, columns=["w_spx","w_10y","w_gold","w_oil"], dtype=float)
for date in trade_test.index:
    node = node_index.loc[date]
    reg  = trade_test.loc[date, "Regime_trade"]
    w_tgt.loc[date] = cache_by_node[node][reg]

# Execution layer (REGIME) — no smoothing
w_path = w_tgt.copy()

# ========= GLOBAL MODEL =========
global_by_node = {}
prev_global = None

for t in rebal_nodes:
    hist_all = trade.loc[:t].iloc[:-1]
    sample = hist_all[asset_lexcess_cols].dropna()
    if len(sample) >= 2:
        mu_g  = sample.mean().values
        Sig_g = sample.cov().values
        anchor = prev_global if prev_global is not None else 0.5*(MIN_W+MAX_W)
        w_g = mv_optimize_box(mu_g, Sig_g, MIN_W, MAX_W,
                              risk_aversion=RISK_AVERSION,
                              anchor=anchor, gamma=GAMMA_STICK)
        prev_global = w_g.copy()
    else:
        if prev_global is not None:
            w_g = prev_global.copy()
        else:
            w_g = mv_optimize_box(mu_train, Sigma_train, MIN_W, MAX_W,
                                  risk_aversion=RISK_AVERSION)
            prev_global = w_g.copy()
    global_by_node[t] = w_g

# Map to daily & execution layer (GLOBAL) — no smoothing
w_tgt_global = pd.DataFrame(index=trade_test.index, columns=["w_spx","w_10y","w_gold","w_oil"], dtype=float)
for date in trade_test.index:
    node = node_index.loc[date]
    w_tgt_global.loc[date] = global_by_node[node]
w_path_global = w_tgt_global.copy()

# ========= Regime vs Global vs Naive — cumulative returns =========
def cum_index(r: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1.0 + r.dropna()).cumprod()

trade_test = trade_test.copy()
trade_test["ret_regime"] = (
    w_path["w_spx"]  * trade_test["ret_spx"]  +
    w_path["w_10y"]  * trade_test["ret_10Y"]  +
    w_path["w_gold"] * trade_test["ret_gold"] +
    w_path["w_oil"]  * trade_test["ret_oil"]
)
trade_test["ret_global"] = (
    w_path_global["w_spx"]  * trade_test["ret_spx"]  +
    w_path_global["w_10y"]  * trade_test["ret_10Y"]  +
    w_path_global["w_gold"] * trade_test["ret_gold"] +
    w_path_global["w_oil"]  * trade_test["ret_oil"]
)
trade_test["ret_naive"] = (
    W_NAIVE[0] * trade_test["ret_spx"] +
    W_NAIVE[1] * trade_test["ret_10Y"] +
    W_NAIVE[2] * trade_test["ret_gold"] +
    W_NAIVE[3] * trade_test["ret_oil"]
)

cum_regime = cum_index(trade_test["ret_regime"])
cum_global = cum_index(trade_test["ret_global"])
cum_naive  = cum_index(trade_test["ret_naive"])

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(cum_regime.index, cum_regime.values, lw=1.8, label="Regime model")
ax.plot(cum_global.index, cum_global.values, lw=1.8, label="Global (no regimes)", alpha=0.9)
ax.plot(cum_naive.index,  cum_naive.values,  lw=1.6, label=f"Naive {W_NAIVE[0]:.0%}/{W_NAIVE[1]:.0%}/{W_NAIVE[2]:.0%}/{W_NAIVE[3]:.0%}", alpha=0.9)
ax.set_title("Cumulative Return — Regime vs Global vs Naive (index = 100)")
ax.set_ylabel("Index level")
ax.grid(alpha=0.25); ax.legend(loc="best")
fig.tight_layout(); plt.show()

# ========= FIGURE: Stacked weights with regime shading (REGIME MODEL) =========
REGIME_COLORS = {
    "Rising–Steepening":   "#1f78b4",
    "Rising–Flattening":   "#ff7f00",
    "Falling–Steepening":  "#33a02c",
    "Falling–Flattening":  "#6a3d9a",
}

def add_regime_annotations(ax, dates, regimes, draw_vlines=True, shade=True, alpha=0.12):
    r = pd.Series(regimes.values, index=dates).dropna()
    if r.empty: return
    flips = r.ne(r.shift(1))
    flip_dates = r.index[flips]
    if draw_vlines:
        for d in flip_dates[1:]:
            ax.axvline(d, linestyle=":", linewidth=0.8, color="k", alpha=0.6, zorder=0)
    if shade:
        seg_starts = list(flip_dates); seg_ends = list(flip_dates[1:]) + [r.index[-1]]
        for s, e in zip(seg_starts, seg_ends):
            ax.axvspan(s, e, facecolor=REGIME_COLORS.get(r.loc[s], "#eeeeee"), alpha=alpha, zorder=0)

fig, ax = plt.subplots(figsize=(10,4))
ax.stackplot(
    w_path.index,
    w_path["w_spx"], w_path["w_10y"], w_path["w_gold"], w_path["w_oil"],
    labels=["SPX","10Y","Gold","Oil"], alpha=0.85
)
ax.set_ylim(0,1.0)
ax.set_title(f"Target weights — Regime model (rebalanced {REBAL_FREQ})")
ax.set_ylabel("Weight"); ax.legend(loc="upper left")
add_regime_annotations(ax, w_path.index, trade_test["Regime_trade"].reindex(w_path.index), True, True, 0.35)
ax.grid(alpha=0.25)
fig.tight_layout(); plt.show()

# ========= FIGURE: GLOBAL stacked weights plot =========
fig, ax = plt.subplots(figsize=(10,4))
ax.stackplot(
    w_path_global.index,
    w_path_global["w_spx"], w_path_global["w_10y"], w_path_global["w_gold"], w_path_global["w_oil"],
    labels=["SPX","10Y","Gold","Oil"], alpha=0.85
)
ax.set_ylim(0,1.0)
ax.set_title(f"Target weights — GLOBAL (no regimes, rebalanced {REBAL_FREQ})")
ax.set_ylabel("Weight"); ax.legend(loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout(); plt.show()

# ========= FIGURE: Regime visualization (Rising/Falling + Steepening/Flattening) =========
fig, ax = plt.subplots(figsize=(12,4.5))
r = trade["Regime"].dropna()
flips = r.ne(r.shift(1))
flip_dates = r.index[flips]
seg_starts = list(flip_dates); seg_ends = list(flip_dates[1:]) + [r.index[-1]]
for s, e in zip(seg_starts, seg_ends):
    lab = r.loc[s]
    ax.axvspan(s, e, facecolor=REGIME_COLORS.get(lab, "#eeeeee"), alpha=0.18, zorder=0)

ax.plot(trade.index, trade["DifY_smooth"], lw=2.0, label="Yield slope: 10Y–3M")
ax.plot(trade.index, trade["AvgY_smooth"], lw=2.0, label="Yield level: AvgY")
ax.axvline(TRAIN_END_DATE, linestyle="--", color="k", lw=1.2, label="TRAIN_END_DATE")

ax.set_title("Regimes based on yield curve dynamics (Rising/Falling × Steepening/Flattening)")
ax.set_ylabel("Percent / percentage points")
ax.grid(alpha=0.25)
line_leg = ax.legend(loc="upper right", ncol=1, frameon=True)  

handles = [Patch(facecolor=REGIME_COLORS[k], alpha=0.18, label=k) for k in REGIMES]
regime_leg = ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
ax.add_artist(line_leg)      # re-add the line legend so both show
ax.add_artist(regime_leg)    # add the regime legend

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout(); plt.show()

# ========= Quick log-excess stats vs RF =========
def ann_stats_from_daily(r: pd.Series):
    r = r.dropna()
    mu_d = r.mean(); sd_d = r.std(ddof=1)
    ann_ret = mu_d * 252.0
    ann_vol = sd_d * np.sqrt(252.0) if pd.notna(sd_d) else np.nan
    ann_sh  = ann_ret / ann_vol if ann_vol else np.nan
    return ann_ret, ann_vol, ann_sh

trade_test["gross_regime"] = 1.0 + trade_test["ret_regime"]
trade_test["gross_global"] = 1.0 + trade_test["ret_global"]
trade_test["gross_naive"]  = 1.0 + trade_test["ret_naive"]
trade_test["gross_rf"]     = 1.0 + trade_test["rf_daily"]

lexcess_regime = np.log(trade_test["gross_regime"]) - np.log(trade_test["gross_rf"])
lexcess_global = np.log(trade_test["gross_global"]) - np.log(trade_test["gross_rf"])
lexcess_naive  = np.log(trade_test["gross_naive"])  - np.log(trade_test["gross_rf"])

a1,v1,s1 = ann_stats_from_daily(lexcess_regime)
aG,vG,sG = ann_stats_from_daily(lexcess_global)
a2,v2,s2 = ann_stats_from_daily(lexcess_naive)
print(f"\nRegime model — AnnLogExcess={a1:.3f}, Vol={v1:.3f}, Sharpe_like={s1:.3f}")
print(f"Global (no regimes) — AnnLogExcess={aG:.3f}, Vol={vG:.3f}, Sharpe_like={sG:.3f}")
print(f"Naive {W_NAIVE[0]:.0%}/{W_NAIVE[1]:.0%}/{W_NAIVE[2]:.1%}/{W_NAIVE[3]:.1%} — AnnLogExcess={a2:.3f}, Vol={v2:.3f}, Sharpe_like={s2:.3f}")

# ========= Performance table by Asset × Regime (TRAIN and ALL) =========
ASSETS = {
    "S&P 500":  {"log": "log_spx",  "lex": "lexcess_spx"},
    "10Y UST":  {"log": "log_10Y",  "lex": "lexcess_10Y"},
    "Gold":     {"log": "log_gold", "lex": "lexcess_gold"},
    "Oil":      {"log": "log_oil",  "lex": "lexcess_oil"},
}
REGIMES = ["Rising–Steepening","Rising–Flattening","Falling–Steepening","Falling–Flattening"]

def _block(sample_df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    def add_rows(sub_df, regime_name):
        for asset, cols in ASSETS.items():
            log_r = sub_df[cols["log"]].dropna()
            if len(log_r) >= 2:
                ann_ret = float(np.exp(log_r.mean()*252.0) - 1.0)
                ann_vol = float(log_r.std(ddof=1)*np.sqrt(252.0))
            else:
                ann_ret, ann_vol = np.nan, np.nan
            sh_ret, sh_vol, sh = ann_stats_from_daily(sub_df[cols["lex"]])
            nobs = int(log_r.shape[0])
            rows.append([label, regime_name, asset, ann_ret, ann_vol, sh, nobs])
    add_rows(sample_df, "All (no split)")
    for r in REGIMES:
        add_rows(sample_df.loc[sample_df["Regime_trade"] == r], r)
    out = pd.DataFrame(rows, columns=["Sample","Regime","Asset",
                                      "Ann. return (%)","Ann. vol (%)",
                                      "Sharpe (excess)","Obs (days)"])
    out["Ann. return (%)"] = 100*out["Ann. return (%)"]
    out["Ann. vol (%)"]    = 100*out["Ann. vol (%)"]
    return out

tbl_all   = _block(trade,       "ALL")
tbl_train = _block(trade_train, "TRAIN")
tbl_test   = _block(trade_test,       "TEST")
perf_tbl  = pd.concat([tbl_all, tbl_train, tbl_test], ignore_index=True)

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\n=== Performance by Asset and Regime ===")
print(perf_tbl.to_string(index=False))

# ========= Sharpe difference test (H1: Regime > Global) =========
def sharpe_diff_test(regime_excess, global_excess):
    x = regime_excess.dropna().values
    y = global_excess.dropna().values
    n = len(x)
    SR_R = x.mean() / x.std(ddof=1)
    SR_G = y.mean() / y.std(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    dS = SR_R - SR_G
    var_diff = (2.0 - 2.0 * rho + 0.5 * (SR_R**2 + SR_G**2) - (SR_R * SR_G)) / n
    se = sqrt(var_diff)
    z = dS / se
    # one-sided p-value: H1 SR_R > SR_G
    p = 1 - norm.cdf(z)
    return {
        "S_regime_daily": SR_R,
        "S_global_daily": SR_G,
        "rho_returns": rho,
        "n": n,
        "Z": z,
        "p_value": p
    }
out = sharpe_diff_test(lexcess_regime, lexcess_global)
print("===\nSharpe difference test (H1: Regime > Global):===")
for k, v in out.items():
    print(f"{k}: {v:.6f}")

# ========= FIGURE: Q–Q plot and ACF of daily excess returns (check assumptions for sharpe difference test)  =========
x = lexcess_regime.dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#  Q–Q plot (Normality assumption) :
stats.probplot(x, dist="norm", plot=axes[0])
axes[0].set_title("Q–Q Plot: Daily Excess Returns")
#  ACF plot (IID assumption) :
plot_acf(x, lags=20, ax=axes[1], alpha=None)  # alpha=None removes default bands
axes[1].axhline(0.1, color="gray", linestyle="dashed", linewidth=0.5)
axes[1].axhline(-0.1, color="gray", linestyle="dashed", linewidth=0.5)
axes[1].set_title("ACF: Daily Excess Returns")
plt.tight_layout()
plt.show()


# ============ FIGURE: Per-regime commulative returns ===========
REGIME_ORDER = ["Rising–Steepening","Rising–Flattening","Falling–Steepening","Falling–Flattening"]

def cum_index_masked(returns: pd.Series, mask: pd.Series, base: float = 1.0) -> pd.Series:
    r_eff = returns.where(mask, 0.0).fillna(0.0)
    return base * (1.0 + r_eff).cumprod()

Z = trade_test[["ret_regime","ret_global","ret_naive","rf_daily","Regime_trade"]].dropna(
    subset=["ret_regime","ret_global","ret_naive","Regime_trade"]
).copy()

series_per_regime = {}
ymin, ymax = np.inf, -np.inf
for r in REGIME_ORDER:
    mask_r = (Z["Regime_trade"] == r)
    idx_reg = cum_index_masked(Z["ret_regime"], mask_r, base=1.0)
    idx_glb = cum_index_masked(Z["ret_global"], mask_r, base=1.0)
    idx_nav = cum_index_masked(Z["ret_naive"],  mask_r, base=1.0)
    series_per_regime[r] = (idx_reg, idx_glb, idx_nav)
    local_min = np.nanmin([idx_reg.min(), idx_glb.min(), idx_nav.min()])
    local_max = np.nanmax([idx_reg.max(), idx_glb.max(), idx_nav.max()])
    ymin = min(ymin, local_min)
    ymax = max(ymax, local_max)

yrange = ymax - ymin
if np.isfinite(yrange) and yrange > 0:
    pad = 0.03 * yrange
    ymin_plot, ymax_plot = ymin - pad, ymax + pad
else:
    ymin_plot, ymax_plot = 0.9, 1.1

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
axes = axes.flatten()
for ax, r in zip(axes, REGIME_ORDER):
    idx_reg, idx_glb, idx_nav = series_per_regime[r]
    ax.plot(idx_reg.index, idx_reg.values, lw=1.8, label="Regime strategy")
    ax.plot(idx_glb.index, idx_glb.values, lw=1.6, label="Global benchmark", alpha=0.9)
    ax.plot(idx_nav.index, idx_nav.values, lw=1.4, label="Naive benchmark",  alpha=0.9)
    ax.set_title(r)
    ax.grid(alpha=0.25)
    ax.set_ylim(ymin_plot, ymax_plot)

fig.suptitle("Cumulative return (index=1) during each regime — TEST period", y=0.98)
for ax in axes[2:]:
    ax.set_xlabel("Date")
for ax in axes[::2]:
    ax.set_ylabel("Index level")
handles, labels = axes[0].get_legend_handles_labels()
fig.subplots_adjust(bottom=0.15, top=0.93, left=0.07, right=0.98)
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False)
plt.show()

# ========== Per-regime Sharpe ratios ==========
def sharpe_ann_from_log_excess(x: pd.Series):
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    mu_d = x.mean()
    sd_d = x.std(ddof=1)
    if not np.isfinite(sd_d) or sd_d == 0:
        return np.nan
    return (mu_d * 252.0) / (sd_d * np.sqrt(252.0))

SR_rows = []
for r in REGIME_ORDER:
    mask_r = (trade_test["Regime_trade"] == r)
    le_R = lexcess_regime[mask_r]
    le_G = lexcess_global[mask_r]
    le_N = lexcess_naive[mask_r]
    SR_R = sharpe_ann_from_log_excess(le_R)
    SR_G = sharpe_ann_from_log_excess(le_G)
    SR_N = sharpe_ann_from_log_excess(le_N)
    SR_rows.append({
        "Regime": r,
        "Sharpe (Regime)": SR_R,
        "Sharpe (Global)": SR_G,
        "Sharpe (Naive)":  SR_N,
        "Obs (days)":      int(mask_r.sum())
    })

regime_sharpes_ts = pd.DataFrame(SR_rows)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\n=== Sharpe ratios by regime (annualized, TEST period) ===")
print(regime_sharpes_ts.to_string(index=False))



