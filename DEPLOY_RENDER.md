# 🚀 Deploy to Render.com (Free Alternative to Railway)

Render offers 750 free hours/month — perfect for your trading bot running 24/7.

---

## 📋 Quick Setup (5 Minutes)

### Step 1: Push render.yaml to GitHub

Your bot now has a `render.yaml` file configured. Push it to GitHub:

```bash
cd C:\Users\HP\Desktop\ai
git add render.yaml
git commit -m "add: Render deployment config"
git push origin main
```

---

### Step 2: Create Render Account

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Click **"Sign in with GitHub"**
4. Authorize Render to access your repositories

---

### Step 3: Create New Service

1. In Render Dashboard, click **"New +"** (top right)
2. Select **"Blueprint"**
3. Connect your repository: **jjgrandma/SMC**
4. Render will detect `render.yaml` automatically
5. Click **"Apply"**

---

### Step 4: Add Secret Environment Variables

Render will create the service but you need to add 3 **secret** variables manually:

1. Click on your **trading-bot** service
2. Go to **"Environment"** tab (left sidebar)
3. Click **"Add Environment Variable"**
4. Add these 3 secrets (use your actual API keys from .env file):

```
Key: TELEGRAM_BOT_TOKEN
Value: <your_telegram_bot_token_from_env_file>

Key: GROQ_API_KEY
Value: <your_groq_api_key_from_env_file>

Key: GEMINI_API_KEY
Value: <your_gemini_api_key_from_env_file>
```

5. Click **"Save Changes"**
6. Render will automatically **redeploy** with the secrets

---

### Step 5: Wait for Deployment

1. Go to **"Logs"** tab
2. Watch the build process (takes 2-3 minutes)
3. Look for these success messages:
   ```
   ✅ Signal scanner started.
   ✅ Keep-alive started — prevents Railway sleep.
   ✅ Scanning GC=F on ['M15', 'H1', 'H4'] for 3 subscribers...
   ✅ yfinance: GC=F W1 — 261 candles
   ```

---

### Step 6: Test Your Bot

Open Telegram and send to your bot:
```
/signal H1
```

You should get a trade signal response! 🎉

---

## 🔄 Auto-Deploy on Git Push

Every time you push to GitHub, Render will automatically rebuild and redeploy:

```bash
git add .
git commit -m "update bot"
git push origin main
```

Wait 2-3 minutes and check Render logs to see the new deployment.

---

## 📊 Monitor Your Bot

### Check if Bot is Running
1. Go to Render Dashboard
2. Look for **green "Live"** badge on your service
3. Click on service → **"Logs"** to see live output

### Check Bot Health
Send to your Telegram bot:
```
/status
```

### View Metrics
Render Dashboard → **"Metrics"** tab shows:
- CPU usage
- Memory usage
- Restart count

---

## ✅ Render Free Tier

**What you get:**
- ✅ 750 hours/month (31 days of 24/7 runtime)
- ✅ Auto-restart on crash
- ✅ Auto-deploy from GitHub
- ✅ Free SSL certificates
- ✅ Logs retention
- ✅ Environment variables

**Limitations:**
- Background workers may spin down after 15 min of no HTTP requests
- **Solution:** Keep-alive is already enabled in your bot to prevent this

---

## 🛑 Stop/Restart the Bot

**Suspend (stop):**
1. Render Dashboard → Your service
2. **"Settings"** tab → Scroll down
3. Click **"Suspend Service"**

**Resume:**
1. Same page → Click **"Resume Service"**

**Manual Restart:**
1. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 🔧 Troubleshooting

### Build fails
- Check **"Logs"** tab for error messages
- Make sure `requirements.txt` has all dependencies
- Try manual redeploy: **"Manual Deploy"** → **"Clear build cache & deploy"**

### Bot starts but no alerts
1. Check `SCANNER_ENABLED=true` in Environment tab
2. Verify `TELEGRAM_ALLOWED_USERS` includes your Telegram ID
3. Look for errors in **"Logs"** tab

### "Service unavailable" error
- Render free tier may take 30-60 seconds to wake up
- Keep-alive prevents this for background workers

### Wrong symbol (XAUUSDm errors)
- Go to **"Environment"** tab
- Find `SYMBOL` variable
- Make sure it's set to `GC=F` (not `XAUUSDm`)
- Click **"Save Changes"** to redeploy

---

## 💰 Upgrade Options

If you exceed 750 hours/month or need faster performance:

**Starter Plan ($7/month):**
- Unlimited hours
- Better CPU/memory
- Priority support

**Current usage:**
- 1 bot running 24/7 = 720 hours/month ✅ (fits free tier)

---

## 🆚 Render vs Railway

| Feature | Render Free | Railway Free |
|---------|-------------|--------------|
| Hours/month | 750 | 500 |
| Auto-deploy | ✅ | ✅ |
| Keep-alive | ✅ | ✅ |
| Build time | 2-3 min | 2-3 min |
| Logs | ✅ | ✅ |

Both work great — Render gives you 50% more free hours!

---

## ✨ Your Bot is Live!

Once deployed on Render, your bot will:
- ✅ Run 24/7 in the cloud
- ✅ Send entry alerts every 15 minutes
- ✅ Auto-restart on crash
- ✅ Stay awake (no sleep)
- ✅ Auto-deploy on git push

**Next:** Push `render.yaml` to GitHub and follow Step 3 above!
