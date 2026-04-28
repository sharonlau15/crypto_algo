# 🚀 Quick Start: Streamlit Dashboard

## 1️⃣ Prerequisites

Your backtest results are already loaded and ready! ✅

```bash
# Verify results exist
ls -la results/
# Should show: strategy_metrics.json, portfolio_returns.csv, etc.
```

## 2️⃣ Launch the Dashboard

### Option A: Fastest Way (Recommended)
```bash
cd /Users/sharonlau15/Desktop/SMU\ 3rd\ Sem/3in1Project/crypto_algo
streamlit run dashboard.py
```

### Option B: Using the Launcher Script
```bash
bash run_dashboard.sh
```

### Option C: With Custom Port
```bash
streamlit run dashboard.py --server.port 8502
```

## 3️⃣ Access the Dashboard

After running one of the commands above:
- **Local URL**: http://localhost:8501
- Dashboard opens automatically in your default browser
- If not, copy the URL from terminal output

## 4️⃣ Navigation Guide

### 📊 Sidebar (Left Side)
- **Select Strategies**: Check/uncheck strategies to compare
- **View Mode**: 4 different analysis views
- Shows last updated time and number of strategies

### 📈 Overview Tab (Default)
**Best for:** Quick performance summary
- Key metrics cards (Sharpe, CAGR, Max DD, Win Rate)
- Cumulative returns chart comparing all strategies
- Performance metrics table
- P&L summary (initial capital, total return, best/worst days)

### 🔍 Individual Analysis Tab
**Best for:** Deep dive into one strategy
- Individual strategy metrics
- Cumulative returns
- Drawdown analysis
- Daily returns histogram
- Rolling Sharpe ratio
- Monthly performance heatmap

### ⚖️ Comparison Tab
**Best for:** Compare multiple strategies side-by-side
- Cumulative returns comparison
- Drawdown comparison
- Distribution of daily returns
- Rolling Sharpe ratio comparison
- Highlighted best/worst metrics

### ⚠️ Risk Analysis Tab
**Best for:** Understanding downside risk
- Drawdown underwater plot
- Rolling 30-day volatility
- Risk metrics summary table

## 5️⃣ Example Workflows

### Workflow 1: Find Your Best Strategy
1. Go to **Overview** tab
2. Look at metrics table sorted by Sharpe Ratio
3. Click on best strategy name to drill down in **Individual Analysis**
4. Review monthly heatmap for consistency

### Workflow 2: Compare Two Strategies
1. Go to **Sidebar** → uncheck all but 2 strategies
2. Go to **Comparison** tab
3. Review all 4 charts
4. Look at metrics table for direct comparison

### Workflow 3: Assess Risk Profile
1. Go to **Risk Analysis** tab
2. Check drawdown history
3. Review rolling volatility
4. Verify Sharpe/Sortino ratios match your risk tolerance

### Workflow 4: Identify Seasonal Patterns
1. Go to **Individual Analysis**
2. Select a strategy
3. Scroll down to "Monthly Performance" heatmap
4. Look for recurring patterns (e.g., better in certain months)

## 6️⃣ Tips & Tricks

### 💡 Performance Tips
| Issue | Solution |
|-------|----------|
| Dashboard slow | Press `R` to refresh, or Press `C` to clear cache |
| Too many charts | Select fewer strategies in sidebar |
| Want to refresh results | Run `python main.py --mode backtest` again, then refresh browser |

### 🎨 Visual Tips
- **Click and drag** on charts to zoom
- **Double-click** on charts to reset zoom
- **Hover** over data points for exact values
- **Highlight/unhighlight** legend items by clicking them

### 📊 Reading the Charts

**Cumulative Returns Chart**:
- Higher line = better performance
- Steep climb = high returns during that period
- Flat line = not trading / no significant moves

**Drawdown Chart**:
- Lower (more negative) = worse drawdown experienced
- Sharp spikes = sudden losses
- Wide base = prolonged underwater period

**Rolling Sharpe Ratio**:
- Higher = better risk-adjusted returns during that period
- Negative = losing money
- Trend upward = improving strategy performance

## 7️⃣ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Rerun script (refresh data) |
| `C` | Clear cache (reload from disk) |
| `Ctrl+C` | Stop dashboard (exit) |
| `?` | Show help menu |

## 8️⃣ Common Questions

### Q: How do I update the dashboard with new results?
**A:** Run a new backtest, then refresh the dashboard:
```bash
python main.py --mode backtest
# Then press R in the dashboard or refresh browser
```

### Q: Can I compare strategies from different backtest runs?
**A:** Currently, the dashboard shows results from your latest backtest. To compare across runs, save results with different names and manually combine CSV files.

### Q: How do I export charts?
**A:** Hover over any Plotly chart → click camera icon (📷) in the toolbar to save as PNG

### Q: Can I use the dashboard while backtesting?
**A:** Yes! The dashboard will show the latest results. Refresh (press `R`) to see updates.

### Q: Why are some metrics N/A?
**A:** Some metrics may not be calculated by your backtest. Check `results/strategy_metrics.json` to verify data is present.

## 9️⃣ Troubleshooting

### Dashboard won't start
```bash
# Check if port 8501 is in use
lsof -i :8501

# Kill process if needed
kill -9 <PID>

# Try different port
streamlit run dashboard.py --server.port 8502
```

### "No backtest results found" error
```bash
# Run backtest first
python main.py --mode backtest

# Verify results exist
ls results/
```

### Charts show "Unable to parse data"
```bash
# Clear cache
# In dashboard: Press C

# Or restart dashboard with cache clearing
streamlit run dashboard.py --logger.level=debug
```

## 🔟 Next Steps

After exploring the dashboard:

1. **Optimize**: Adjust strategy parameters and re-backtest
2. **Deploy**: Run live trading with `python main.py --mode live`
3. **Monitor**: Check dashboard daily for live P&L
4. **Extend**: Add custom metrics to `dashboard.py`

## 📚 More Information

- Full documentation: See `DASHBOARD.md`
- Code reference: See `dashboard.py` for customization options
- Streamlit docs: https://docs.streamlit.io/

---

**Ready to explore?**
```bash
streamlit run dashboard.py
```

Dashboard opens at: **http://localhost:8501** 🎉
