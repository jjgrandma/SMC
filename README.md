# 🤖 Gold SMC Trading Bot

An agentic AI Forex trading system for **XAUUSD (Gold)** built with Smart Money Concepts, dual AI (Gemini + Groq), and MetaTrader 5 integration.

## Features

- 📊 Full SMC analysis — BOS, CHoCH, FVG, Order Blocks, Liquidity
- 🔄 Multi-timeframe cascade: W1 → D1 → H4 → H1 → M15
- 🤖 Dual AI engine — Gemini (deep) + Groq (fast), best result chosen
- 📋 Manual trade signal cards with full reasoning
- 🌅 Daily morning briefing at London open
- 💰 Per-user risk management (lot sizing by account balance)
- 📉 Performance dashboard with pip tracking
- 🧠 Self-learning memory system
- 🖥 MetaTrader 5 integration (live price + execution)
- 🔔 Auto-alerts via Telegram scanner

## Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env with your keys
```

### 4. Run
```bash
# Terminal 1 — API
python run_api.py

# Terminal 2 — Bot
python run_bot.py
```

## Required API Keys

| Key | Where to get | Cost |
|-----|-------------|------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free |
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) | Free |

## Bot Commands

| Command | Description |
|---------|-------------|
| `/trade [TF]` | Manual signal card with full SMC reasoning |
| `/signal [TF]` | Quick trade signal |
| `/analyze [TF]` | Full SMC analysis |
| `/briefing` | Morning briefing + day plan |
| `/swing` | Swing trade idea |
| `/performance` | Win rate, pip P&L, stats dashboard |
| `/history` | Trade history list |
| `/memory` | Bot learning stats |
| `/outcome` | Mark trade WIN/LOSS |
| `/mt5connect` | Connect MetaTrader 5 |

## Project Structure

```
app/
├── agent.py          # AI decision engine
├── config.py         # Settings from .env
├── dual_ai.py        # Gemini + Groq parallel engine
├── journal.py        # Trade journal with pip tracking
├── main.py           # FastAPI backend
├── memory.py         # Self-learning memory system
├── mtf_analysis.py   # Multi-timeframe cascade
├── pip_utils.py      # Pip calculations
├── risk.py           # Position sizing
├── scanner.py        # Auto-alert background scanner
├── smc_adapter.py    # smartmoneyconcepts library adapter
├── smc_engine.py     # Built-in SMC engine
├── tools.py          # Market data (MT5 → yfinance → fallback)
├── trader.py         # MetaTrader 5 execution layer
└── user_profile.py   # Per-user settings store
bot/
└── telegram_bot.py   # Telegram bot UI
```

## Deployment

See [Railway](#railway) or [VPS](#vps) sections below.

### Railway (recommended — free tier available)

1. Push to GitHub
2. Go to [railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Add environment variables from `.env`
5. Deploy

### VPS (Ubuntu)

```bash
# Install
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
cp .env.example .env && nano .env

# Run with screen
screen -S api
python run_api.py
# Ctrl+A, D

screen -S bot
python run_bot.py
# Ctrl+A, D
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
AI_PROVIDER=groq
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_ALLOWED_USERS=your_telegram_id
SYMBOL=XAUUSDm
```

## License

MIT
