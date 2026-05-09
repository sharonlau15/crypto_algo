"""Quick diagnostics — run this on the server to see what's failing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.client import get_client

client = get_client(for_trading=True)

print("=== futures_symbol_ticker ===")
try:
    tickers = client.futures_symbol_ticker()
    btc = next((t for t in tickers if t["symbol"] == "BTCUSDT"), None)
    print(f"BTC price: {btc}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== futures_account ===")
try:
    acc = client.futures_account()
    print(f"totalWalletBalance : {acc.get('totalWalletBalance')}")
    print(f"totalUnrealizedProfit: {acc.get('totalUnrealizedProfit')}")
    assets = acc.get("assets", [])
    print(f"assets count       : {len(assets)}")
    for a in assets:
        if float(a.get("walletBalance", 0)) > 0:
            print(f"  {a['asset']}: walletBalance={a['walletBalance']}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== futures_change_leverage (BTCUSDT, 1x) ===")
try:
    r = client.futures_change_leverage(symbol="BTCUSDT", leverage=1)
    print(f"OK: {r}")
except Exception as e:
    print(f"FAILED: {e}")
