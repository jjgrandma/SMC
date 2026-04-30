"""
Professional UI components for the Telegram bot.
All formatting, keyboards, and visual elements in one place.
"""

from __future__ import annotations
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Color / emoji system  (consistent across all messages)
# ---------------------------------------------------------------------------

class Icon:
    # Directions
    BUY       = "🟢"
    SELL      = "🔴"
    NEUTRAL   = "⚪"
    # Outcomes
    WIN       = "✅"
    LOSS      = "❌"
    PENDING   = "⏳"
    BREAKEVEN = "➖"
    CANCEL    = "🚫"
    # Confidence
    HIGH      = "🔥"
    MEDIUM    = "⚡"
    LOW       = "💡"
    # Zones
    PREMIUM   = "🔻"
    DISCOUNT  = "🔺"
    EQ        = "⚖️"
    # UI
    BACK      = "◀️"
    HOME      = "🏠"
    REFRESH   = "🔄"
    CHART     = "📊"
    SIGNAL    = "⚡"
    TRADE     = "📋"
    BRIEFING  = "🌅"
    SWING     = "📈"
    STATUS    = "📡"
    PERF      = "📉"
    HISTORY   = "🗂"
    MEMORY    = "🧠"
    PROFILE   = "👤"
    ALERT     = "🔔"
    MT5       = "🖥"
    MONEY     = "💰"
    RISK      = "⚠️"
    LOCK      = "🔒"
    CLOCK     = "🕐"
    CALENDAR  = "📅"
    TARGET    = "🎯"
    STOP      = "🛑"
    TP        = "✅"
    RR        = "📐"
    LOT       = "📦"
    PIP       = "📏"
    STAR      = "⭐"
    FIRE      = "🔥"
    BOLT      = "⚡"
    BRAIN     = "🧠"
    ROBOT     = "🤖"
    GOLD      = "🥇"
    UP        = "📈"
    DOWN      = "📉"
    NEWS      = "📰"
    WARN      = "⚠️"
    OK        = "✅"
    ERROR     = "❌"
    INFO      = "ℹ️"
    SETTINGS  = "⚙️"
    SEARCH    = "🔍"
    LOADING   = "⏳"


# ---------------------------------------------------------------------------
# Dividers and section headers
# ---------------------------------------------------------------------------

DIV  = "━━━━━━━━━━━━━━━━━━━━━━━━"
DIV2 = "─────────────────────────"
DIV3 = "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"


def header(title: str, subtitle: str = "") -> str:
    lines = [f"*{title}*"]
    if subtitle:
        lines.append(f"_{subtitle}_")
    lines.append(DIV)
    return "\n".join(lines)


def section(title: str) -> str:
    return f"\n*{title}*"


def field(label: str, value: str, icon: str = "") -> str:
    prefix = f"{icon} " if icon else "  "
    return f"{prefix}{label}: `{value}`"


def badge(text: str, icon: str = "") -> str:
    return f"{icon} `{text}`" if icon else f"`{text}`"


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------

def bar_blocks(value: float, total: float = 100, width: int = 10) -> str:
    """Filled block bar: ██████░░░░"""
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, int((value / total) * width)))
    return "█" * filled + "░" * (width - filled)


def bar_squares(wins: int, losses: int, width: int = 10) -> str:
    """Green/red square bar: 🟩🟩🟩🟥🟥"""
    total = wins + losses
    if total == 0:
        return "⬜" * width
    w = max(0, min(width, int(wins / total * width)))
    l = width - w
    return "🟩" * w + "🟥" * l


def trend_arrow(direction: str) -> str:
    if direction == "bullish":
        return "↗️"
    if direction == "bearish":
        return "↘️"
    return "↔️"


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------

def kb(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    """
    Build keyboard from list of rows.
    Each row is a list of (label, callback_data) tuples.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=cb) for label, cb in row]
        for row in rows
    ])


def back_kb(extra_rows: list[list[tuple[str, str]]] | None = None) -> InlineKeyboardMarkup:
    """Keyboard with optional extra rows + back button."""
    rows = extra_rows or []
    rows.append([(f"{Icon.BACK} Back to Menu", "main_menu")])
    return kb(rows)


def main_menu_kb() -> InlineKeyboardMarkup:
    return kb([
        [(f"{Icon.CHART} Analyze H1",  "analyze_H1"),
         (f"{Icon.CHART} Analyze H4",  "analyze_H4")],
        [(f"{Icon.SIGNAL} Signal H1",  "signal_H1"),
         (f"{Icon.SIGNAL} Signal H4",  "signal_H4")],
        [(f"{Icon.TRADE} Trade Card",  "trade_H1"),
         (f"{Icon.SWING} Swing",       "swing")],
        [(f"{Icon.BRIEFING} Briefing", "briefing"),
         (f"{Icon.STATUS} Status",     "status")],
        [(f"{Icon.PERF} Performance",  "performance"),
         (f"{Icon.HISTORY} History",   "history")],
        [(f"{Icon.MEMORY} Memory",     "memory"),
         (f"{Icon.PROFILE} Profile",   "profile")],
        [(f"{Icon.ALERT} Alerts",      "toggle_alerts"),
         (f"{Icon.MT5} MT5",           "mt5_status")],
    ])


def signal_action_kb(action: str, tf: str, symbol: str) -> InlineKeyboardMarkup:
    """Keyboard shown after a signal — refresh + back."""
    return back_kb([
        [(f"{Icon.REFRESH} Refresh Signal", f"signal_{tf}"),
         (f"{Icon.TRADE} Trade Card",       f"trade_{tf}")],
    ])


def analysis_action_kb(tf: str) -> InlineKeyboardMarkup:
    return back_kb([
        [(f"{Icon.SIGNAL} Get Signal",  f"signal_{tf}"),
         (f"{Icon.REFRESH} Refresh",    f"analyze_{tf}")],
    ])


def timeframe_kb(callback_prefix: str) -> InlineKeyboardMarkup:
    return back_kb([
        [("M15", f"{callback_prefix}_M15"), ("H1", f"{callback_prefix}_H1"),
         ("H4", f"{callback_prefix}_H4"),  ("D1", f"{callback_prefix}_D1")],
    ])


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def fmt_loading(action: str, symbol: str, tf: str = "") -> str:
    tf_str = f" `{tf}`" if tf else ""
    return f"{Icon.LOADING} _{action} {symbol}{tf_str}..._"


def fmt_error(message: str) -> str:
    return f"{Icon.ERROR} *Error*\n\n_{message}_"


def fmt_no_trade(reason: str, news_blocked: bool = False) -> str:
    lines = [
        f"{Icon.NEUTRAL} *NO TRADE*",
        DIV,
        f"_{reason}_",
    ]
    if news_blocked:
        lines += ["", f"{Icon.WARN} _High-impact news window active_"]
    return "\n".join(lines)


def fmt_signal(data: dict, risk_result=None, profile=None) -> str:
    action     = data.get("action", "N/A")
    symbol     = data.get("symbol", "XAUUSD")
    tf         = data.get("timeframe", "H1")
    price_info = data.get("current_price", {})
    conf       = data.get("confidence", "N/A")
    pd_zone    = data.get("premium_discount", "N/A")
    htf_bias   = data.get("htf_bias", "N/A")

    action_icon = Icon.BUY if action == "BUY" else Icon.SELL
    conf_icon   = {"HIGH": Icon.HIGH, "MEDIUM": Icon.MEDIUM, "LOW": Icon.LOW}.get(conf, "")
    pd_icon     = Icon.PREMIUM if pd_zone == "premium" else Icon.DISCOUNT if pd_zone == "discount" else Icon.EQ
    bias_icon   = Icon.BUY if htf_bias == "bullish" else Icon.SELL if htf_bias == "bearish" else Icon.NEUTRAL

    entry = data.get("entry", "—")
    sl    = data.get("stop_loss", "—")
    tp    = data.get("take_profit", "—")
    rr    = data.get("rr_ratio", "—")

    lines = [
        f"{action_icon} *{action}* — {symbol} `{tf}`",
        DIV,
        f"{Icon.CLOCK} `{price_info.get('timestamp','')[:16]} UTC`",
        f"{Icon.MONEY} Price: `{price_info.get('mid','—')}` _{price_info.get('source','')}_",
        f"{conf_icon} Confidence: *{conf}*",
        f"{pd_icon} Zone: *{pd_zone.upper()}*  {bias_icon} HTF: *{htf_bias.upper()}*",
        "",
        f"*Entry Levels*",
        f"  {Icon.TARGET} Entry:  `{entry}`",
        f"  {Icon.STOP}  SL:     `{sl}`",
        f"  {Icon.TP}    TP:     `{tp}`",
        f"  {Icon.RR}    R:R:    `1:{rr}`",
    ]

    # Key levels
    kl = data.get("key_levels", {})
    if kl:
        lines += [
            "",
            f"*Key Levels*",
            f"  Support:    `{kl.get('support','—')}`",
            f"  Resistance: `{kl.get('resistance','—')}`",
        ]

    # Risk block
    if risk_result and risk_result.approved and profile:
        from app.pip_utils import price_to_pips
        sl_pips = price_to_pips(risk_result.pip_risk, symbol)
        tp_pips = price_to_pips(abs(float(tp or 0) - float(entry or 0)), symbol)
        profit  = risk_result.risk_amount * risk_result.rr_ratio
        lines += [
            "",
            f"*Risk Management*",
            f"  {Icon.MONEY} Balance:  `${profile.account_balance:,.0f}` @ `{profile.risk_percent}%`",
            f"  {Icon.RISK}  Risk $:   `${risk_result.risk_amount:,.2f}`",
            f"  {Icon.LOT}   Lots:     `{risk_result.lot_size}`",
            f"  {Icon.PIP}   SL:       `{sl_pips:.0f} pips`",
            f"  {Icon.PIP}   TP:       `{tp_pips:.0f} pips`",
            f"  {Icon.TARGET} Reward:  `${profit:,.2f}`",
        ]
    elif risk_result and not risk_result.approved:
        lines += ["", f"{Icon.WARN} _Risk: {risk_result.rejection_reason}_"]

    # Confluences
    confluences = data.get("confluences", [])
    if confluences:
        lines += ["", f"*Confluences* ({len(confluences)})"]
        for c in confluences[:5]:
            lines.append(f"  {Icon.OK} _{c}_")

    # Reasoning
    reasoning = data.get("reasoning", "")
    if reasoning:
        lines += ["", f"*Reasoning*", f"_{reasoning[:300]}_"]

    # Invalidation
    inv = data.get("invalidation", "")
    if inv:
        lines += ["", f"*Invalidation*", f"_{inv}_"]

    # AI comparison
    ai_cmp = data.get("ai_comparison", {})
    if ai_cmp and ai_cmp.get("chosen") not in (None, "none"):
        g = ai_cmp.get("gemini", {})
        q = ai_cmp.get("groq", {})
        chosen = ai_cmp.get("chosen", "").upper()
        lines += [
            "",
            DIV2,
            f"{Icon.ROBOT} *AI* — Chosen: *{chosen}*",
            f"  🔬 Gemini: `{g.get('action','?')}` score=`{g.get('score','?')}` `{g.get('latency_ms','?')}ms`",
            f"  ⚡ Groq:   `{q.get('action','?')}` score=`{q.get('score','?')}` `{q.get('latency_ms','?')}ms`",
        ]

    lines += ["", DIV, f"_{data.get('symbol','XAUUSD')} • {tf} • SMC Analysis_"]

    full = "\n".join(lines)
    return full[:4000] + "\n_...truncated_" if len(full) > 4000 else full


def fmt_analysis(data: dict) -> str:
    mtf   = data.get("mtf_data", {})
    price = data.get("current_price", {})
    sym   = data.get("symbol", "XAUUSD")
    tf    = data.get("timeframe", "H1")

    w_bias = mtf.get("weekly_bias", "ranging")
    d_bias = mtf.get("daily_bias", "ranging")
    aligned = mtf.get("htf_aligned", False)
    overall = mtf.get("overall_bias", "ranging")
    confs   = mtf.get("confluence_count", 0)
    allowed = mtf.get("trade_allowed", False)

    w_icon = Icon.BUY if w_bias == "bullish" else Icon.SELL if w_bias == "bearish" else Icon.NEUTRAL
    d_icon = Icon.BUY if d_bias == "bullish" else Icon.SELL if d_bias == "bearish" else Icon.NEUTRAL
    o_icon = Icon.BUY if overall == "bullish" else Icon.SELL if overall == "bearish" else Icon.NEUTRAL
    align_str = f"{Icon.OK} Aligned" if aligned else f"{Icon.WARN} Not Aligned"
    trade_str = f"{Icon.OK} Trade Allowed" if allowed else f"{Icon.LOCK} No Trade"

    lines = [
        f"{Icon.CHART} *SMC ANALYSIS — {sym} `{tf}`*",
        DIV,
        f"{Icon.MONEY} `{price.get('mid','—')}` _{price.get('source','')}_",
        f"{Icon.NEWS} News: `{'BLOCKED ⚠️' if data.get('news_blocked') else 'Clear'}`",
        "",
        f"*Higher Timeframe Bias*",
        f"  {w_icon} Weekly: *{w_bias.upper()}*",
        f"  {d_icon} Daily:  *{d_bias.upper()}*",
        f"  {o_icon} Overall: *{overall.upper()}*",
        f"  {align_str}  •  {trade_str}",
        f"  {Icon.BOLT} Confluences: `{confs}`",
    ]

    # Per-timeframe table
    tfs_data = mtf.get("timeframes", {})
    if tfs_data:
        lines += ["", f"*Timeframe Breakdown*"]
        for tf_key in ["W1", "D1", "H4", "H1", "M15", "M5"]:
            if tf_key not in tfs_data:
                continue
            t = tfs_data[tf_key]
            trend = t.get("trend", "ranging")
            t_icon = Icon.BUY if trend == "bullish" else Icon.SELL if trend == "bearish" else Icon.NEUTRAL
            pd = t.get("premium_discount", "eq")
            pd_icon = Icon.PREMIUM if pd == "premium" else Icon.DISCOUNT if pd == "discount" else Icon.EQ
            fvg = t.get("active_fvg", 0)
            ob  = t.get("active_ob", 0)
            lines.append(
                f"  `{tf_key:3}` {t_icon} {trend[:4]:4} {pd_icon} "
                f"FVG:`{fvg}` OB:`{ob}`"
            )

    # AI narrative
    narrative = data.get("analysis", "")
    if narrative:
        lines += ["", f"*AI Analysis*", narrative[:1500]]

    lines += ["", DIV, f"_{sym} • SMC Multi-Timeframe_"]

    full = "\n".join(lines)
    return full[:4000] + "\n_...truncated_" if len(full) > 4000 else full


def fmt_performance(stats) -> str:
    wr      = stats.win_rate
    wr_icon = Icon.BUY if wr >= 60 else "🟡" if wr >= 45 else Icon.SELL
    total_d = stats.wins + stats.losses + stats.breakevens

    streak_icon = Icon.FIRE if stats.current_streak_type == "WIN" else "❄️" if stats.current_streak_type == "LOSS" else ""
    streak_str  = f"{streak_icon} `{stats.current_streak}` {stats.current_streak_type}" if stats.current_streak else "—"

    last10 = ""
    for t in stats.last_10:
        last10 += Icon.WIN if t["outcome"] == "WIN" else Icon.LOSS if t["outcome"] == "LOSS" else Icon.BREAKEVEN
    last10 = last10 or "_No completed trades_"

    net_sign = "+" if stats.total_pips >= 0 else ""
    net_icon = Icon.UP if stats.total_pips >= 0 else Icon.DOWN

    lines = [
        f"{Icon.PERF} *PERFORMANCE DASHBOARD*",
        DIV,
        f"*Results*",
        f"  Total:     `{stats.total}` signals",
        f"  Completed: `{total_d}`  Pending: `{stats.pending}`",
        "",
        f"  {bar_squares(stats.wins, stats.losses)}",
        f"  {Icon.WIN} Wins:   `{stats.wins}` ({wr}%)",
        f"  {Icon.LOSS} Losses: `{stats.losses}` ({stats.loss_rate}%)",
        f"  {Icon.BREAKEVEN} BE:  `{stats.breakevens}`",
        "",
        f"  {wr_icon} Win Rate: `{wr}%`  `{bar_blocks(wr)}`",
        "",
        f"*Pip P&L*",
        f"  {net_icon} Net:      `{net_sign}{stats.total_pips} pips`",
        f"  {Icon.WIN} Won:      `+{stats.pips_won} pips`",
        f"  {Icon.LOSS} Lost:     `{stats.pips_lost} pips`",
        f"  Avg win:  `+{stats.avg_pips_win} pips`",
        f"  Avg loss: `{stats.avg_pips_loss} pips`",
        f"  Best:     `+{stats.best_trade_pips} pips`",
        "",
        f"*Risk/Reward*",
        f"  Target:   `1:{stats.avg_rr_target}`",
        f"  Achieved: `1:{stats.avg_rr_achieved}`",
        f"  Best:     `1:{stats.best_rr}`",
        "",
        f"*Streaks*",
        f"  Current:  {streak_str}",
        f"  Best run: `{stats.best_win_streak}` wins",
        f"  Worst:    `{stats.worst_loss_streak}` losses",
        "",
        f"*Last 10*  {last10}",
    ]

    # Breakdowns
    if stats.by_timeframe:
        lines += ["", f"*By Timeframe*"]
        for tf, d in sorted(stats.by_timeframe.items(), key=lambda x: -x[1]["win_rate"]):
            icon = Icon.BUY if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else Icon.SELL
            lines.append(f"  {icon} `{tf}` {d['wins']}W/{d['losses']}L = `{d['win_rate']}%` `{bar_blocks(d['win_rate'])}`")

    if stats.by_session:
        lines += ["", f"*By Session*"]
        for sess, d in sorted(stats.by_session.items(), key=lambda x: -x[1]["win_rate"]):
            icon = Icon.BUY if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else Icon.SELL
            lines.append(f"  {icon} `{sess}` {d['wins']}W/{d['losses']}L = `{d['win_rate']}%`")

    if stats.by_confidence:
        lines += ["", f"*By Confidence*"]
        for conf, d in sorted(stats.by_confidence.items(), key=lambda x: -x[1]["win_rate"]):
            icon = {"HIGH": Icon.HIGH, "MEDIUM": Icon.MEDIUM, "LOW": Icon.LOW}.get(conf, "")
            lines.append(f"  {icon} `{conf}` {d['wins']}W/{d['losses']}L = `{d['win_rate']}%`")

    if stats.by_day:
        lines += ["", f"*By Day*"]
        for day in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
            if day in stats.by_day:
                d = stats.by_day[day]
                icon = Icon.BUY if d["win_rate"] >= 60 else "🟡" if d["win_rate"] >= 45 else Icon.SELL
                lines.append(f"  {icon} `{day[:3]}` {d['wins']}W/{d['losses']}L = `{d['win_rate']}%`")

    lines += ["", DIV, f"_/history • /outcome • /performance 30_"]

    full = "\n".join(lines)
    return full[:4000] + "\n_...truncated_" if len(full) > 4000 else full


def fmt_profile(profile) -> str:
    conf_icon = {"HIGH": Icon.HIGH, "MEDIUM": Icon.MEDIUM, "LOW": Icon.LOW}.get(profile.min_confidence, "")
    alert_str = f"{Icon.OK} ON" if profile.alerts_enabled else f"{Icon.ERROR} OFF"
    max_risk  = profile.account_balance * profile.risk_percent / 100

    return (
        f"{Icon.PROFILE} *YOUR PROFILE*\n"
        f"{DIV}\n"
        f"{Icon.MONEY} Balance:    `${profile.account_balance:,.2f}`\n"
        f"{Icon.RISK}  Risk/Trade: `{profile.risk_percent}%` = `${max_risk:,.2f}`\n"
        f"{Icon.CHART} Default TF: `{profile.timeframe}`\n"
        f"{Icon.ALERT} Alerts:     {alert_str}\n"
        f"{conf_icon} Min Conf:   `{profile.min_confidence}`\n"
        f"{DIV}\n"
        f"_/setbalance • /setrisk • /settf • /alerts_"
    )


def fmt_status(data: dict) -> str:
    price   = data.get("current_price", {})
    trades  = data.get("active_trades", [])
    account = data.get("account", {})
    symbol  = data.get("symbol", "XAUUSD")

    lines = [
        f"{Icon.STATUS} *LIVE STATUS — {symbol}*",
        DIV,
        f"{Icon.MONEY} `{price.get('mid','—')}` _{price.get('source','')}_",
    ]

    if account and "balance" in account:
        mode = account.get("trade_mode_name", "")
        mode_icon = "🟡" if mode == "DEMO" else "🔴" if mode == "REAL" else ""
        lines += [
            "",
            f"*MT5 Account* {mode_icon} `{mode}`",
            f"  Balance: `${account.get('balance',0):,.2f}`",
            f"  Equity:  `${account.get('equity',0):,.2f}`",
            f"  Margin:  `${account.get('margin',0):,.2f}`",
        ]

    if trades:
        lines += ["", f"*Open Positions ({len(trades)})*"]
        for t in trades:
            direction = "BUY" if t.get("type") == 0 else "SELL"
            pnl       = t.get("profit", 0)
            pnl_icon  = Icon.WIN if pnl >= 0 else Icon.LOSS
            d_icon    = Icon.BUY if direction == "BUY" else Icon.SELL
            lines.append(
                f"  {pnl_icon} #{t.get('ticket')} {d_icon} `{direction}` "
                f"{t.get('volume')} lots @ `{t.get('price_open')}` "
                f"P&L: `${pnl:.2f}`"
            )
    else:
        lines += ["", f"_No open positions_"]

    lines += ["", DIV]
    return "\n".join(lines)


def fmt_history(records: list) -> str:
    if not records:
        return f"{Icon.HISTORY} *TRADE HISTORY*\n{DIV}\n_No trades yet._"

    lines = [f"{Icon.HISTORY} *TRADE HISTORY* (last {len(records)})", DIV]

    for r in records:
        outcome_icon = {
            "WIN": Icon.WIN, "LOSS": Icon.LOSS,
            "PENDING": Icon.PENDING, "BREAKEVEN": Icon.BREAKEVEN,
        }.get(r.outcome, Icon.CANCEL)

        d_icon = Icon.BUY if r.action == "BUY" else Icon.SELL

        if r.outcome in ("WIN", "LOSS", "BREAKEVEN") and r.outcome_pips != 0:
            pip_str = f"`{'+' if r.outcome_pips >= 0 else ''}{r.outcome_pips:.0f}p`"
        else:
            pip_str = f"TP:`{r.tp_pips:.0f}p`"

        rr_str = f"`{r.outcome_rr_achieved}RR`" if r.outcome_rr_achieved else f"`{r.rr_ratio}RR`"

        lines.append(
            f"{outcome_icon} {d_icon} `{r.action}` `{r.timeframe}` "
            f"@ `{r.entry}` {pip_str} {rr_str} "
            f"`{r.session[:3]}` `{r.timestamp[:10]}`"
        )
        if r.outcome_note:
            lines.append(f"   _{r.outcome_note}_")

    lines += ["", DIV, f"_/performance • /outcome_"]
    full = "\n".join(lines)
    return full[:4000] + "\n_...truncated_" if len(full) > 4000 else full


def fmt_main_menu(symbol: str, profile) -> str:
    return (
        f"{Icon.ROBOT} *SMC GOLD TRADING BOT*\n"
        f"{DIV}\n"
        f"{Icon.GOLD} `{symbol}`  "
        f"{Icon.MONEY} `${profile.account_balance:,.0f}`  "
        f"{Icon.RISK} `{profile.risk_percent}%`\n"
        f"{DIV2}\n"
        f"_Select an option below_"
    )
