import time
import hmac
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Load keys from Binance.env
env_path = Path(__file__).parent / "Binance.env"
load_dotenv(env_path)

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")

BASE_URL = "https://testnet.binance.vision"

def signed_request(endpoint, params=None):
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    signature = hmac.new(
        API_SECRET.encode(),
        query_string.encode(),
        hashlib.sha256,
    ).hexdigest()
    url = f"{BASE_URL}{endpoint}?{query_string}&signature={signature}"
    return requests.get(url, headers={"X-MBX-APIKEY": API_KEY})


# ── Fetch all prices (no auth needed) ─────────────────────────────────────────
price_resp = requests.get(f"{BASE_URL}/api/v3/ticker/price")
prices = {p["symbol"]: float(p["price"]) for p in price_resp.json()} if price_resp.ok else {}

def to_usdt(asset: str, qty: float) -> float:
    if asset == "USDT":
        return qty
    return qty * prices.get(f"{asset}USDT", 0.0)


# ── Account balances ───────────────────────────────────────────────────────────
resp = signed_request("/api/v3/account")

if resp.status_code != 200:
    print(f"Error {resp.status_code}: {resp.text}")
else:
    data = resp.json()
    balances = [b for b in data["balances"] if float(b["free"]) > 0 or float(b["locked"]) > 0]

    print(f"\n{'Asset':<10} {'Free':>18} {'Locked':>18} {'Value (USDT)':>16}")
    print("-" * 66)
    total_usdt = 0.0
    for b in balances:
        free   = float(b["free"])
        locked = float(b["locked"])
        value  = to_usdt(b["asset"], free + locked)
        total_usdt += value
        print(f"{b['asset']:<10} {free:>18.6f} {locked:>18.6f} {value:>16.2f}")

    print("-" * 66)
    print(f"{'TOTAL':<10} {'':<18} {'':<18} {total_usdt:>16.2f}")
    print(f"\nTotal assets with balance: {len(balances)}")
