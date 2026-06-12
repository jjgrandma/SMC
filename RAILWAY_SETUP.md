# Railway Deployment Guide

## ✅ What Was Fixed

1. **Fixed entry alerts** — 4 bugs blocking signals:
   - Syntax error in `agent.py` (bot wouldn't start)
   - `MAX_SL_POINTS = 35` → `100` (SL too tight for Gold)
   - `MIN_CONFLUENCES = 3` → `2` (gate too strict)
   - Fake economic calendar removed
   - Missing `get_pending_signals()` method added

2. **Added 24/7 keep-alive** — prevents Railway from sleeping:
   - `app/keepalive.py` — pings health endpoint every 10 minutes
   - `railway.toml` — sets `sleepApplication = false`

3. **Fixed symbol** — `XAUUSDm` → `GC=F` (Gold Futures for yfinance)

4. **Disabled MT5** — no longer tries to connect to MetaTrader 5

---

## 🚀 Railway Deployment Steps

### 1. Push to GitHub
```bash
git push origin main
```
✅ Already done — commit `68a86e8` is live.

### 2. Railway Auto-Deploy
If you have **auto-deploy enabled** in Railway:
- Railway will automatically pull the latest code
- Rebuild and redeploy
- Wait 2-3 minutes and check logs

If **auto-deploy is OFF**:
1. Go to [railway.app](https://railway.app)
2. Find your project
3. Click on the **bot** service
4. Click **"Deploy"** or **"Redeploy"**

### 3. Environment Variables
Make sure these are set in Railway dashboard:

**Required:**
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
TELEGRAM_ALLOWED_USERS=your_telegram_user_id,another_user_id
```

**Important:**
```
SYMBOL=GC=F
MT5_ENABLED=false
SCANNER_ENABLED=true
SCANNER_INTERVAL_MINUTES=15
SCANNER_TIMEFRAMES=M15,H1,H4
```

**Optional (use defaults if not set):**
```
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-1.5-flash
MAX_RISK_PERCENT=1.0
MIN_RR_RATIO=2.0
BRIEFING_HOUR=7
BRIEFING_MINUTE=0
```

---

## 🔍 Verify Deployment

### Check Railway Logs
Look for these success messages:
```
✅ Signal scanner started.
✅ Price alert watcher started.
✅ Morning briefing scheduler started.
✅ Keep-alive started — prevents Railway sleep.
✅ Scanning GC=F on ['M15', 'H1', 'H4'] for X subscribers...
✅ yfinance: GC=F W1 — 261 candles
```

**Red flags (errors):**
- ❌ `MT5 initialize failed` — normal if MT5_ENABLED=false
- ❌ `XAUUSDm: possibly delisted` — means SYMBOL is still set to XAUUSDm instead of GC=F
- ❌ `'MemoryStore' object has no attribute 'get_pending_signals'` — old code, redeploy needed

### Test the Bot
Send to your Telegram bot:
```
/signal H1
```

You should get a signal response (BUY/SELL/NO_TRADE).

---

## 🛠 Railway Config Files

**`railway.toml`** (NEW):
- `sleepApplication = false` — keeps bot running 24/7
- `restartPolicyType = "ON_FAILURE"` — auto-restart on crash

**`Procfile`**:
```
bot: python run_bot.py
```

**`runtime.txt`** (if exists):
```
python-3.11.x
```

---

## 📊 Keep-Alive Mechanism

**How it works:**
- `app/keepalive.py` pings `API_BASE_URL/health` every 10 minutes
- If API is deployed on Railway too, set `API_BASE_URL` to the Railway API URL
- If API is NOT deployed, the bot will run standalone (doesn't need API)

**Check if keep-alive is working:**
Look for this in logs every 10 minutes:
```
Keep-alive ping successful: {'status': 'ok', 'symbol': 'GC=F', ...}
```

---

## 🔥 Quick Fixes

### Bot still not sending alerts?
1. Check Railway logs for errors
2. Verify `SCANNER_ENABLED=true`
3. Check `TELEGRAM_ALLOWED_USERS` includes your Telegram ID
4. Test with `/signal H1` command manually

### Bot keeps sleeping?
1. Check `railway.toml` is in the repo
2. Verify keep-alive logs show pings every 10 minutes
3. Railway free tier may still have limits — upgrade to Hobby plan if needed

### Symbol errors?
Make sure Railway environment has:
```
SYMBOL=GC=F
```
NOT `XAUUSDm` or `XAUUSD`.

---

## 📞 Support

If issues persist:
1. Check Railway deployment logs
2. Verify all environment variables are set
3. Test locally first: `python run_bot.py`
4. Make sure commit `68a86e8` or later is deployed

**Current fixes pushed:**
- ✅ Commit `4b2c9e3` — entry alert fixes
- ✅ Commit `68a86e8` — Railway keep-alive + memory fix
