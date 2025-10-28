# NSE Factor Explorer (Streamlit)

Interactive factor backtesting for NIFTY stocks:
- Factors: Momentum (12m, skip 1m), Low Volatility (1/σ_60d), Trend (50/200 MA spread)
- Rebalance: Monthly (business month-end)
- Universe: NIFTY 50 (editable)
- Benchmark: NIFTY 50 Index (^NSEI)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
