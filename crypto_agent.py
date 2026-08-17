from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os, json, requests, warnings, logging
import pandas as pd
import ta
import json as json_lib
from pathlib import Path

TRADE_LOG_FILE = "trade_history.json"
COOLDOWN_CYCLES = 3  # number of recent runs a symbol stays "cooling down" after a trade

def load_trade_history():
    if Path(TRADE_LOG_FILE).exists():
        with open(TRADE_LOG_FILE) as f:
            return json_lib.load(f)
    return []

def save_trade(symbol, action):
    history = load_trade_history()
    history.append({
        "symbol": symbol,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    with open(TRADE_LOG_FILE, "w") as f:
        json_lib.dump(history[-50:], f, indent=2)  # keep last 50 trades only

def recently_traded(symbol):
    history = load_trade_history()
    recent = history[-COOLDOWN_CYCLES:]
    return any(t["symbol"] == symbol for t in recent)

warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR)

load_dotenv(override=True)

alpaca_key = os.getenv("ALPACA_API_KEY")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
data_client = CryptoHistoricalDataClient(alpaca_key, alpaca_secret)
trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True)
cryptopanic_token = os.getenv("CRYPTOPANIC_TOKEN")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "AVAX/USD", "LINK/USD", "UNI/USD"]

STOP_LOSS_PCT = -5.0        # auto-sell if a position drops below this % P/L
MAX_ALLOCATION_PCT = 15.0   # don't buy more of a symbol if it already exceeds this % of portfolio

# ---------- STEP 1: hard-coded stop-loss, runs BEFORE any LLM call ----------
def enforce_stop_loss():
    positions = trading_client.get_all_positions()
    for p in positions:
        pl_pct = float(p.unrealized_plpc) * 100
        if pl_pct <= STOP_LOSS_PCT:
            print(f"🛑 STOP-LOSS TRIGGERED: {p.symbol} at {pl_pct:.2f}% — selling automatically.")
            order_request = MarketOrderRequest(
                symbol=p.symbol,
                qty=abs(float(p.qty)),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_request)

enforce_stop_loss()


def get_technicals(symbol):
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=datetime.now(timezone.utc) - timedelta(days=30)  # extended to 30d for range context
    )
    bars = data_client.get_crypto_bars(request).df
    if bars.empty or len(bars) < 30:
        return None
    close = bars['close']
    volume = bars['volume']

    def pct_change(hours):
        if len(close) < hours:
            return None
        return round(((close.iloc[-1] - close.iloc[-hours]) / close.iloc[-hours]) * 100, 2)

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
    sma_short = close.rolling(10).mean().iloc[-1]
    sma_long = close.rolling(30).mean().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct = ((close.iloc[-1] - bb.bollinger_lband().iloc[-1]) /
              (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]))
    volume_trend = "rising" if volume.iloc[-5:].mean() > volume.iloc[-20:-5].mean() else "falling"

    # NEW: position within 30-day range
    range_high = close.max()
    range_low = close.min()
    range_position = (close.iloc[-1] - range_low) / (range_high - range_low)  # 0 = at 30d low, 1 = at 30d high

    # NEW: z-score vs 30-day mean
    z_score = (close.iloc[-1] - close.mean()) / close.std()

    return {
        "change_1h": pct_change(1), "change_4h": pct_change(4),
        "change_24h": pct_change(24), "change_7d": pct_change(min(len(close)-1, 168)),
        "rsi": round(rsi, 1),
        "macd_signal": "bullish" if macd_diff > 0 else "bearish",
        "trend": "uptrend" if sma_short > sma_long else "downtrend",
        "bollinger_position": round(bb_pct, 2),
        "volume_trend": volume_trend,
        "range_position_30d": round(range_position, 2),   # 0=near 30d low, 1=near 30d high
        "z_score_30d": round(z_score, 2)                  # >1.5 = extended/overbought territory, <-1.5 = oversold
    }

# ---------- STEP 2: technicals (unchanged) ----------
def get_technicals(symbol):
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=datetime.now(timezone.utc) - timedelta(days=7)
    )
    bars = data_client.get_crypto_bars(request).df
    if bars.empty or len(bars) < 30:
        return None
    close = bars['close']
    volume = bars['volume']

    def pct_change(hours):
        if len(close) < hours:
            return None
        return round(((close.iloc[-1] - close.iloc[-hours]) / close.iloc[-hours]) * 100, 2)

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
    sma_short = close.rolling(10).mean().iloc[-1]
    sma_long = close.rolling(30).mean().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct = ((close.iloc[-1] - bb.bollinger_lband().iloc[-1]) /
              (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]))
    volume_trend = "rising" if volume.iloc[-5:].mean() > volume.iloc[-20:-5].mean() else "falling"

    return {
        "change_1h": pct_change(1), "change_4h": pct_change(4),
        "change_24h": pct_change(24), "change_7d": pct_change(len(close) - 1),
        "rsi": round(rsi, 1),
        "macd_signal": "bullish" if macd_diff > 0 else "bearish",
        "trend": "uptrend" if sma_short > sma_long else "downtrend",
        "bollinger_position": round(bb_pct, 2),
        "volume_trend": volume_trend
    }

def get_news_sentiment(symbol):
    coin = symbol.split("/")[0]
    if not cryptopanic_token:
        return None
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={cryptopanic_token}&currencies={coin}&public=true"
        posts = requests.get(url, timeout=5).json().get("results", [])[:5]
        return "; ".join(p["title"] for p in posts) if posts else "No recent news"
    except Exception:
        return "News unavailable"

# ---------- STEP 3: full holdings data with real P/L ----------
account = trading_client.get_account()
positions = trading_client.get_all_positions()
portfolio_value = float(account.portfolio_value)

held_symbols = {}
if positions:
    lines = []
    for p in positions:
        pl_pct = float(p.unrealized_plpc) * 100
        allocation_pct = (float(p.market_value) / portfolio_value) * 100
        held_symbols[p.symbol] = allocation_pct
        lines.append(
            f"{p.symbol}: {p.qty} units, value ${p.market_value}, "
            f"unrealized P/L {pl_pct:+.2f}% (${p.unrealized_pl}), "
            f"{allocation_pct:.1f}% of portfolio"
        )
    positions_text = "\n".join(lines)
else:
    positions_text = "No current positions."

# ---------- STEP 4: build analysis, respecting position limits ----------
request_params = CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
quotes = data_client.get_crypto_latest_quote(request_params)

blocks = []
buyable_symbols = []
for sym in symbols:
    q = quotes[sym]
    tech = get_technicals(sym)
    news = get_news_sentiment(sym)
    current_allocation = held_symbols.get(sym, 0)
    at_limit = current_allocation >= MAX_ALLOCATION_PCT

    if not at_limit:
        buyable_symbols.append(sym)

    if tech:
        block = (
            f"{sym}:\n"
            f"  Price - Bid: {q.bid_price}, Ask: {q.ask_price}\n"
            f"  Change - 1h: {tech['change_1h']}%, 4h: {tech['change_4h']}%, "
            f"24h: {tech['change_24h']}%, 7d: {tech['change_7d']}%\n"
            f"  RSI(14): {tech['rsi']}, MACD: {tech['macd_signal']}, Trend: {tech['trend']}\n"
            f"  Bollinger position: {tech['bollinger_position']}, Volume: {tech['volume_trend']}\n"
            f"  Current allocation: {current_allocation:.1f}%"
            f"{' — AT LIMIT, cannot buy more' if at_limit else ''}\n"
            f"  Recent news: {news}"
        )
    else:
        block = f"{sym}: insufficient historical data"
    blocks.append(block)

analysis_text = "\n\n".join(blocks)

# symbols the LLM is allowed to choose for buying = not at limit; for selling, any held symbol is fine
allowed_symbols = list(set(buyable_symbols) | set(held_symbols.keys()))

trade_tool = {
    "type": "function",
    "function": {
        "name": "make_trade_decisions",
        "description": "Decide on a set of trade actions across multiple symbols.",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "enum": symbols},
                            "action": {"type": "string", "enum": ["buy", "sell"]},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["symbol", "action", "reasoning"]
                    }
                }
            },
            "required": ["decisions"]
        }
    }
}

prompt = f"""Your current cash: ${account.cash}
Portfolio value: ${portfolio_value}

Your current positions (review these for potential exits, not just new buys):
{positions_text}

Full market analysis:
{analysis_text}

Rules already enforced automatically (do not violate):
- Any position below {STOP_LOSS_PCT}% P/L is already auto-sold before you see this.
- You cannot buy more of a symbol already at or above {MAX_ALLOCATION_PCT}% portfolio allocation.

You are a professional trader, not a retail chaser. For each symbol, explicitly reason about:
1. Has this move already happened (near 30d high, high z-score) — in which case buying now is
   chasing, likely low reward for the risk — versus is there still room to run?
2. Is a dip a genuine opportunity (temporarily oversold, fundamentals/trend intact) or a falling
   knife (downtrend accelerating, no reversal signal)?
3. Do NOT buy simply because price is going up — momentum alone is not a reason. Only act when
   trend, volume, and position-in-range align in the same direction.

Actively consider selling an existing position if its trend/indicators have turned unfavorable,
not just scanning for new buys. You may take multiple actions this cycle if warranted — e.g.,
sell a losing position AND buy a strong opportunity in the same response. Only include symbols
you want to act on (buy or sell); omit anything you want to hold.

Current allocation: {current_allocation:.1f}% — {'❌ HARD BLOCKED: DO NOT propose buying this symbol, you are already at the maximum allocation' if at_limit else 'OK to buy'}
Use the make_trade_decisions tool to respond."""

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": prompt}],
    tools=[trade_tool],
    temperature=0.3,
)


message = response.choices[0].message

def normalize_symbol(raw_symbol, valid_symbols):
    """Map a possibly-malformed symbol (e.g. 'BTCUSD') back to the canonical form ('BTC/USD')."""
    cleaned = raw_symbol.replace("/", "").replace("-", "").upper()
    for s in valid_symbols:
        if s.replace("/", "").upper() == cleaned:
            return s
    return None  # no match found

if message.tool_calls:
    result = json.loads(message.tool_calls[0].function.arguments)
    decisions = result.get("decisions", [])

    if not decisions:
        print("⏸️  No actions — holding across the board.")

    for decision in decisions:
        raw_symbol = decision["symbol"]
        symbol = normalize_symbol(raw_symbol, symbols)
        action = decision["action"]

        if symbol is None:
            print(f"⚠️ Unrecognized symbol from LLM: {raw_symbol} — skipping.")
            continue

        if recently_traded(symbol):
            print(f"⏳ Skipping {symbol} — recently traded, in cooldown to avoid whipsaw.")
            continue

        print(f"\nSymbol: {symbol}")
        print(f"Action: {action}")
        print(f"Reasoning: {decision['reasoning']}")

        if action == "buy" and symbol not in buyable_symbols:
            print(f"⚠️ Blocked: {symbol} already at allocation limit.")
            continue

        order_side = OrderSide.BUY if action == "buy" else OrderSide.SELL
        if order_side == OrderSide.SELL:
            held_qty = next((p.qty for p in positions if p.symbol == symbol), None)
            if held_qty is None:
                print(f"⚠️ Blocked: no holding found for {symbol}, cannot sell.")
                continue
            order_request = MarketOrderRequest(symbol=symbol, qty=abs(float(held_qty)),
                                                 side=order_side, time_in_force=TimeInForce.GTC)
        else:
            order_request = MarketOrderRequest(symbol=symbol, notional=1000,
                                                 side=order_side, time_in_force=TimeInForce.GTC)

        order = trading_client.submit_order(order_request)
        print(f"✅ Order placed: {action.upper()} {symbol} — status: {order.status}")
        save_trade(symbol, action)
else:
    print("No tool call made. Raw response:", message.content)