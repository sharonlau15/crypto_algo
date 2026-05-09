#!/bin/bash
# run_dashboard.sh
# Streamlit dashboard launcher for crypto strategy analysis

set -e

echo "🚀 Starting Crypto Strategy Dashboard..."
echo ""

# Check if results directory has data
if [ ! -d "results" ] || [ ! -f "results/strategy_metrics.json" ]; then
    echo "⚠️  No backtest results found."
    echo "   Run backtest first: python main.py --mode backtest"
    exit 1
fi

# Install streamlit if not present
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit==1.31.1
fi

echo "✅ Starting dashboard on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run dashboard/dashboard.py
