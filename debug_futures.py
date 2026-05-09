"""Quick diagnostics — run this on the server to see what's failing."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Show what keys are actually loaded ────────────────────────────────────────
from dotenv import load_dotenv
for p in [Path("Binance.env"), Path(".env")]:
    if p.exists():
        load_dotenv(p)
        print(f"Loaded env from: {p}")
        break

demo_key    = os.getenv("BINANCE_DEMO_API_KEY", "")
demo_secret = os.getenv("BINANCE_DEMO_API_SECRET", "")
live_key    = os.getenv("BINANCE_API_KEY", "")

print(f"\nBINANCE_DEMO_API_KEY    : '{demo_key[:8]}...{demo_key[-4:]}' (len={len(demo_key)})")
print(f"BINANCE_DEMO_API_SECRET : len={len(demo_secret)}, starts='{demo_secret[:4]}...'")
print(f"BINANCE_API_KEY (live)  : '{live_key[:8]}...' (len={len(live_key)})")
print(f"Demo key == Live key?   : {demo_key == live_key}")

from binance.client import Client

print("\n=== futures_account (demo keys → live fapi.binance.com) ===")
try:
    c = Client(demo_key, demo_secret)
    acc = c.futures_account()
    print(f"totalWalletBalance: {acc.get('totalWalletBalance')}")
    for a in acc.get("assets", []):
        if float(a.get("walletBalance", 0)) > 0:
            print(f"  {a['asset']}: {a['walletBalance']}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== futures_account (demo keys → testnet.binancefuture.com) ===")
try:
    c2 = Client(demo_key, demo_secret, testnet=True)
    acc2 = c2.futures_account()
    print(f"totalWalletBalance: {acc2.get('totalWalletBalance')}")
    for a in acc2.get("assets", []):
        if float(a.get("walletBalance", 0)) > 0:
            print(f"  {a['asset']}: {a['walletBalance']}")
except Exception as e:
    print(f"FAILED: {e}")
