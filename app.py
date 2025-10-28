# app.py
# NSE Factor Explorer — full Streamlit app with rebalance history + colored deltas
# Run: streamlit run app.py

import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="NSE Factor Explorer", layout="wide")

# ---------- Constants ----------
DEFAULT_START = "2018-01-01"
INDEX_TICKER = "^NSEI"  # NIFTY 50 index on yfinance
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","ITC","LT","SBIN","BHARTIARTL","AXISBANK",
    "HINDUNILVR","BAJFINANCE","KOTAKBANK","HCLTECH","MARUTI","SUNPHARMA","NTPC","ASIANPAINT","ULTRACEMCO","TITAN",
    "TATASTEEL","ONGC","POWERGRID","NESTLEIND","M&M","WIPRO","ADANIENT","ADANIPORTS","TATAMOTORS","COALINDIA",
    "HINDZINC","HEROMOTOCO","BAJAJFINSV","GRASIM","JSWSTEEL","DIVISLAB","DRREDDY","BRITANNIA","BPCL","EICHERMOT",
    "INDUSINDBK","TATACONSUM","UPL","APOLLOHOSP","CIPLA","SBILIFE","HDFCLIFE","HINDALCO","TECHM","SHREECEM"
]
# Convert to Yahoo Finance tickers
NIFTY50_YF = [t + ".NS" for t in NIFTY50]

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    """Download adjusted close prices from yfinance (daily)."""
    # yfinance returns DataFrame with columns possibly as a MultiIndex when multiple tickers
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # drop columns with all-NaNs
    df = df.dropna(axis=1, how="all")
    # ensure datetime index and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def pct_returns(prices):
    return prices.pct_change().dropna(how="all")

def monthly_business_ends(idx):
    # month business end dates within the index range; then intersect to ensure trading days
    bm = pd.date_range(idx.min(), idx.max(), freq="BM")
    return bm.intersection(idx)

def momentum_signal(prices, lookback_days=252, skip_days=21):
    # Standard: 12m momentum, skip most recent 1m
    past = prices.shift(skip_days)
    mom = past / past.shift(lookback_days - skip_days) - 1.0
    return mom

def low_vol_signal(rets, window=60):
    vol = rets.rolling(window).std()
    inv_vol = 1 / vol.replace(0, np.nan)
    return inv_vol

def trend_signal(prices, fast=50, slow=200):
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma - slow_ma) / slow_ma  # normalized MA spread

def zscore_last_row(df_row):
    s = df_row.copy().dropna()
    if s.empty or s.std(ddof=0) == 0:
        # preserve index but zeros
        return pd.Series(0.0, index=df_row.index)
    zs = (s - s.mean()) / s.std(ddof=0)
    # reindex to original and fill missing with 0
    return zs.reindex(df_row.index).fillna(0.0)

def annualize_return(daily):
    if len(daily) == 0:
        return np.nan
    return (1 + daily).prod() ** (252 / len(daily)) - 1

def sharpe(daily, rf=0.0):
    ex = daily - rf/252
    if ex.std() == 0:
        return 0.0
    return np.sqrt(252) * ex.mean() / ex.std()

def max_dd(cum):
    return (cum / cum.cummax() - 1).min()

def safe_market_cap_lookup(ticker):
    """Try multiple places to get a market cap; return NaN if not found."""
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None) or {}
        mc = fi.get("market_cap") if isinstance(fi, dict) else None
        if mc is None:
            info = tk.info if hasattr(tk, "info") else {}
            mc = info.get("marketCap", info.get("market_cap", None))
        if mc is None:
            return np.nan
        return mc
    except Exception:
        return np.nan

# ---------- UI ----------
st.title("NSE Factor Explorer — Momentum • Low Vol • Trend")
st.caption("Interactive factor backtesting on NIFTY stocks (yfinance). Built for Yale MAM portfolio-rigor demos.")

with st.sidebar:
    st.header("Configuration")
    universe_choice = st.selectbox("Universe", ["NIFTY 50"], index=0)
    start = st.date_input("Start date", pd.to_datetime(DEFAULT_START).date())
    end = st.date_input("End date", date.today())
    top_n = st.slider("Top-N selection", 5, 20, 10, 1)
    rebalance = st.selectbox("Rebalance frequency", ["Monthly (BM)"], index=0)

    st.subheader("Factor Weights")
    w_mom = st.slider("Momentum weight", 0.0, 1.0, 0.33, 0.01)
    w_lv  = st.slider("Low Volatility weight", 0.0, 1.0, 0.33, 0.01)
    w_tr  = st.slider("Trend weight", 0.0, 1.0, 0.34, 0.01)
    total = w_mom + w_lv + w_tr
    if abs(total - 1.0) > 1e-6:
        st.error("Factor weights should sum to 1.0")
    cap_weight = st.checkbox("Cap-weight within Top-N (proxy for liquidity)", value=True)
    st.markdown("---")
    st.caption("Tip: Keep fast MA 50 and slow MA 200 for classic trend.")
    fast = st.number_input("Trend fast MA (days)", 10, 250, 50, 5)
    slow = st.number_input("Trend slow MA (days)", 20, 400, 200, 10)
    st.markdown("---")
    run = st.button("Run Backtest")

if run:
    # Universe
    tickers = NIFTY50_YF

    # Data
    prices = load_prices(tickers, str(start), str(end))
    if prices.empty:
        st.error("No price data downloaded for the chosen date range / universe.")
        st.stop()

    # Ensure no duplicate timestamps in prices index
    if prices.index.has_duplicates:
        st.warning("Duplicate timestamps found in price data; dropping duplicates (keeping first).")
        prices = prices[~prices.index.duplicated(keep='first')]

    # NOTE: compute returns after final prices are determined (important)
    rets = pct_returns(prices)

    # Load index series and align safely to prices
    index_px = load_prices([INDEX_TICKER], str(start), str(end))
    if index_px.empty:
        st.error(f"No index data for {INDEX_TICKER}. Try a different date range.")
        st.stop()
    # index_px may be a DataFrame with one col; take the first column
    if isinstance(index_px, pd.DataFrame):
        index_px = index_px.iloc[:, 0]

    # Normalize indexes
    index_px.index = pd.to_datetime(index_px.index)
    prices.index = pd.to_datetime(prices.index)

    # Remove duplicates if any
    if index_px.index.has_duplicates:
        st.warning("Duplicate timestamps found in index series; dropping duplicates (keeping first).")
        index_px = index_px[~index_px.index.duplicated(keep='first')]

    # Calculate common index and align robustly
    common_idx = prices.index.intersection(index_px.index)

    if len(common_idx) == 0:
        st.error("No overlapping dates between asset prices and index series. Try a different date range.")
        st.stop()

    if len(common_idx) < len(prices.index):
        missing_count = len(prices.index) - len(common_idx)
        st.warning(f"Index series missing {missing_count} date(s) present in asset prices. Aligning to common dates (nearest/ffill fallback).")

        # Attempt nearest mapping within tolerance; if not possible, reindex then ffill/bfill
        try:
            # tolerance 7 calendar days
            index_px_aligned = index_px.reindex(prices.index, method='nearest', tolerance=pd.Timedelta('7D'))
        except Exception:
            index_px_aligned = index_px.reindex(prices.index)

        # forward/backfill to eliminate NaNs (safer for index level usage)
        index_px_aligned = index_px_aligned.ffill().bfill()

        # Use intersection of available dates after mapping:
        mapped_common = prices.index.intersection(index_px_aligned.index)

        # Restrict prices to mapped_common (these are trading days we will use)
        prices = prices.loc[mapped_common]
        # Recompute returns based on the new prices index (this avoids index/returns misalignment)
        rets = pct_returns(prices)

        # Now align index_px to same mapped_common
        index_px = index_px_aligned.loc[mapped_common]
    else:
        # perfect match -> just align
        index_px = index_px.loc[prices.index]
        # rets already computed above is fine

    index_ret = index_px.pct_change().dropna()

    # Factor signals (time-series)
    mom_ts = momentum_signal(prices)
    lv_ts  = low_vol_signal(rets)
    tr_ts  = trend_signal(prices, fast=fast, slow=slow)

    # Rebalance dates
    bm = monthly_business_ends(prices.index)
    if bm.empty:
        st.error("No business-month-end rebalance dates in the selected window. Try a larger date range.")
        st.stop()

    # Backtest
    port_daily = []
    holdings_rec = []  # structure: [(rebalance_date, selected_list, {ticker: weight%}), ...]
    for i, d in enumerate(bm):
        # skip if rebalance date not in signal index (rare after alignment)
        if d not in mom_ts.index or d not in lv_ts.index or d not in tr_ts.index:
            continue

        hist_rets = rets.loc[:d]
        # Need sufficient history
        if len(hist_rets) < 252 + 21:
            continue

        # Cross-section z-scores at rebalance date
        mom_cs = mom_ts.loc[d].dropna()
        lv_cs  = lv_ts.loc[d].dropna()
        tr_cs  = tr_ts.loc[d].dropna()

        names = sorted(set(prices.columns) & set(mom_cs.index) & set(lv_cs.index) & set(tr_cs.index))
        if len(names) < top_n + 2:
            # not enough names with full signals; skip
            continue

        mom_z = zscore_last_row(mom_cs[names])
        lv_z  = zscore_last_row(lv_cs[names])
        tr_z  = zscore_last_row(tr_cs[names])

        comp = pd.DataFrame({"mom": mom_z, "lv": lv_z, "tr": tr_z})
        comp["score"] = w_mom * comp["mom"] + w_lv * comp["lv"] + w_tr * comp["tr"]
        selected = comp["score"].nlargest(top_n).index.tolist()

        # Weights: either equal within Top-N or cap-weight proxy using market-cap via yfinance
        if cap_weight:
            caps = {}
            for t in selected:
                yf_ticker = t if t.endswith(".NS") else t + ".NS"
                mc = safe_market_cap_lookup(yf_ticker)
                caps[t] = mc
            caps_series = pd.Series(caps)
            # fallback: replace NaNs with median (so we don't blow up)
            if caps_series.isna().any():
                median_cap = caps_series.median()
                caps_series = caps_series.fillna(median_cap if not np.isnan(median_cap) else 1.0)
            w = caps_series / caps_series.sum()
        else:
            w = pd.Series(1/len(selected), index=selected)

        holdings_rec.append((d.date(), selected, (w*100).round(2).to_dict()))

        # Compute returns until next rebalance
        if i < len(bm)-1:
            nxt = bm[i+1]
            sl = rets.loc[d:nxt].iloc[1:]  # start AFTER rebalance day
        else:
            sl = rets.loc[d:].iloc[1:]

        if not sl.empty:
            sub = sl[selected].copy()
            wnorm = w / w.sum()
            port_ret = sub.mul(wnorm, axis=1).sum(axis=1)
            port_daily.append(port_ret)

    if not port_daily:
        st.error("No results. Try expanding the date range, lowering Top-N, or checking data availability.")
        st.stop()

    port_daily = pd.concat(port_daily).sort_index()

    # Align to index returns (bench)
    common = port_daily.index.intersection(index_ret.index)
    if len(common) == 0:
        st.error("No overlapping dates between strategy returns and benchmark returns after processing.")
        st.stop()
    port_daily = port_daily.loc[common]
    bench_daily = index_ret.loc[common]

    # Metrics
    port_cum = (1 + port_daily).cumprod()
    bench_cum = (1 + bench_daily).cumprod()

    met = {
        "Portfolio Ann. Return": f"{100*annualize_return(port_daily):.2f}%",
        "Portfolio Sharpe": f"{sharpe(port_daily):.2f}",
        "Portfolio Max Drawdown": f"{100*max_dd(port_cum):.2f}%",
        "NIFTY50 Ann. Return": f"{100*annualize_return(bench_daily):.2f}%",
        "NIFTY50 Sharpe": f"{sharpe(bench_daily):.2f}",
        "NIFTY50 Max Drawdown": f"{100*max_dd(bench_cum):.2f}%"
    }

    # ---------- Layout ----------
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Performance vs. NIFTY 50")
        fig, ax = plt.subplots(figsize=(9,4))
        port_cum.plot(ax=ax, label="Strategy")
        bench_cum.plot(ax=ax, label="NIFTY 50")
        ax.set_ylabel("Growth of ₹1")
        ax.set_xlabel("")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.subheader("Key Metrics")
        st.table(pd.DataFrame(met, index=["Value"]).T)

    # ---------- Latest holdings ----------
    st.subheader("Latest Rebalance Holdings & Weights")
    if holdings_rec:
        last_date, names, weights = holdings_rec[-1]
        st.write(f"**As of:** {last_date}")
        df_hold = pd.DataFrame({"Ticker": list(weights.keys()), "Weight %": list(weights.values())}).sort_values("Weight %", ascending=False)
        st.dataframe(df_hold, use_container_width=True)
    else:
        st.info("No holdings recorded.")

    # ---------- Rebalance history table with colored deltas ----------
    # Build a rebalance history table from holdings_rec
    if holdings_rec:
        # Build DataFrame of weights: rows = tickers, cols = rebalance dates
        dates = [str(x[0]) for x in holdings_rec]  # e.g. '2023-03-31'
        weight_dicts = [x[2] for x in holdings_rec]  # list of {ticker: weight%}

        # union of all tickers across rebalance history (sorted)
        all_tickers = sorted({t for wd in weight_dicts for t in wd.keys()})

        # weights_df: index = ticker, cols = date strings, values = weight % (float)
        weights_df = pd.DataFrame(index=all_tickers, columns=dates, dtype=float)
        for d, wd in zip(dates, weight_dicts):
            for t, w in wd.items():
                weights_df.at[t, d] = float(w)  # ensure floats
        weights_df = weights_df.fillna(0.0)

        # compute deltas = current_col - previous_col (month-to-month)
        deltas_df = weights_df.diff(axis=1).fillna(0.0)

        # Build combined display DataFrame with interleaved columns: [date1_w, date1_Δ, date2_w, date2_Δ, ...]
        display_df = pd.DataFrame(index=weights_df.index)
        for col in weights_df.columns:
            wcol = f"{col} • Wt%"
            dcol = f"{col} • Δ"
            display_df[wcol] = weights_df[col].round(2)
            display_df[dcol] = deltas_df[col].round(2)

        # Format: positive deltas -> green, negative -> red
        def color_delta(val):
            try:
                v = float(val)
            except Exception:
                return ""
            if v > 0:
                return "color: green; font-weight: 600"
            elif v < 0:
                return "color: red; font-weight: 600"
            else:
                return ""

        # Apply styling to delta columns
        delta_cols = [c for c in display_df.columns if "• Δ" in c]
        styler = display_df.style.format("{:.2f}")\
                                 .applymap(color_delta, subset=delta_cols)\
                                 .set_caption("Rebalance history — weights (%) and month-to-month Δ (green ↑, red ↓)")

        st.subheader("Rebalance History (weights and changes)")
        st.write("This table shows each rebalance date's weight (%) and the change vs previous rebalance.")
        st.write(styler)  # Streamlit renders pandas Styler

    else:
        st.info("No rebalance history available.")

    st.caption("Educational use only. No investment advice. Data source: Yahoo Finance (yfinance).")
else:
    st.info("Configure options in the sidebar and click **Run Backtest**.")
