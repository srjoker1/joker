"""
ارکستریتور اصلی
۴ ایجنت رو اجرا می‌کنه، امتیازها رو ترکیب می‌کنه، نقاط ورود/حد ضرر/حد سود
رو با ATR محاسبه می‌کنه و فقط وقتی سیگنال واقعی و «جدید» باشه به تلگرام می‌فرسته.
"""

import os
import json
import datetime
import traceback
import requests

from agents import FundamentalAgent, SilverIndustrialAgent, TechnicalAgent, CalendarAgent

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]

ACTIONABLE_THRESHOLD = 20
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
COOLDOWN_HOURS = 3
STATE_FILE = os.path.join("state", "last_signal.json")


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=20)
    if not r.ok:
        print("خطا در ارسال پیام تلگرام:", r.text)
    return r.ok


def classify(total_score):
    if total_score >= 50:
        return "🟢 سیگنال خرید قوی (BUY)", "buy"
    elif total_score >= ACTIONABLE_THRESHOLD:
        return "🟢 سیگنال خرید محتاطانه (weak BUY)", "buy"
    elif total_score <= -50:
        return "🔴 سیگنال فروش قوی (SELL)", "sell"
    elif total_score <= -ACTIONABLE_THRESHOLD:
        return "🔴 سیگنال فروش محتاطانه (weak SELL)", "sell"
    else:
        return "⚪️ خنثی / منتظر بمانید (NO TRADE)", "none"


def trade_levels(direction, price, atr_val):
    if direction == "buy":
        return {
            "entry": price,
            "stop_loss": price - SL_ATR_MULT * atr_val,
            "take_profit": price + TP_ATR_MULT * atr_val,
        }
    elif direction == "sell":
        return {
            "entry": price,
            "stop_loss": price + SL_ATR_MULT * atr_val,
            "take_profit": price - TP_ATR_MULT * atr_val,
        }
    return None


def format_levels_line(symbol_label, levels, decimals=2):
    if levels is None:
        return f"• {symbol_label}: بدون معامله (سیگنال خنثی)"
    return (
        f"• {symbol_label} → ورود: {levels['entry']:.{decimals}f} | "
        f"حد ضرر: {levels['stop_loss']:.{decimals}f} | "
        f"حد سود: {levels['take_profit']:.{decimals}f}"
    )


def load_last_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(direction, score):
    os.makedirs("state", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "direction": direction,
                "score": score,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            },
            f,
        )


def should_send(direction, last_state):
    if last_state is None:
        return True
    if last_state.get("direction") != direction:
        return True
    last_time = datetime.datetime.fromisoformat(last_state["timestamp"])
    elapsed_hours = (datetime.datetime.utcnow() - last_time).total_seconds() / 3600
    return elapsed_hours >= COOLDOWN_HOURS


def build_report():
    fundamental = FundamentalAgent()
    silver_agent = SilverIndustrialAgent()
    technical = TechnicalAgent()
    calendar = CalendarAgent()

    fund_score, fund_reason = fundamental.analyze(TWELVEDATA_API_KEY)
    silver_score, silver_reason = silver_agent.analyze(TWELVEDATA_API_KEY)
    tech_score, tech_reason, levels_data = technical.analyze(TWELVEDATA_API_KEY)
    cal_risk, cal_reason = calendar.analyze()

    results = [
        {"agent": fundamental.name, "score": fund_score, "reason": fund_reason},
        {"agent": silver_agent.name, "score": silver_score, "reason": silver_reason},
        {"agent": technical.name, "score": tech_score, "reason": tech_reason},
    ]

    weights = {fundamental.name: 0.3, silver_agent.name: 0.2, technical.name: 0.5}
    total_score = sum(r["score"] * weights.get(r["agent"], 0.33) for r in results)
    total_score = max(-100, min(100, total_score))

    verdict, direction = classify(total_score)
    if cal_risk >= 40:
        verdict += " ⚠️ (احتیاط: نزدیک رویداد کلان مهم)"

    gold_levels = trade_levels(direction, levels_data["XAU/USD"]["price"], levels_data["XAU/USD"]["atr"])
    silver_levels = trade_levels(direction, levels_data["XAG/USD"]["price"], levels_data["XAG/USD"]["atr"])

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"📊 <b>سیگنال طلا/نقره</b> — {now}")
    lines.append(f"\n<b>نتیجه نهایی:</b> {verdict}")
    lines.append(f"<b>امتیاز ترکیبی:</b> {total_score:.1f} / 100")

    lines.append("\n<b>نقاط معامله:</b>")
    lines.append(format_levels_line("طلا (XAU/USD)", gold_levels, decimals=2))
    lines.append(format_levels_line("نقره (XAG/USD)", silver_levels, decimals=3))

    lines.append("\n<b>جزئیات هر ایجنت:</b>")
    for r in results:
        lines.append(f"• <b>{r['agent']}</b> (امتیاز {r['score']}): {r['reason']}")
    lines.append(f"\n<b>تقویم/ریسک رویداد:</b> {cal_reason}")
    lines.append(
        "\n⚠️ حد ضرر و حد سود بر اساس ATR (میانگین دامنه نوسان واقعی) محاسبه شده، "
        "نه توصیه مالی قطعی. همیشه حجم معامله رو متناسب با ریسک‌پذیری خودتون تنظیم کنید."
    )

    return "\n".join(lines), total_score, direction


def main():
    try:
        report_text, total_score, direction = build_report()
    except Exception as e:
        error_text = (
            "⚠️ <b>خطا در اجرای ربات سیگنال</b>\n"
            f"سیستم نتونست تحلیل رو کامل کنه: {e}\n"
            "لطفاً کلیدهای API و توکن‌ها رو بررسی کنید."
        )
        print(traceback.format_exc())
        send_telegram_message(error_text)
        return

    if direction == "none":
        print("سیگنال خنثی بود، پیامی ارسال نشد. امتیاز:", total_score)
        return

    last_state = load_last_state()
    if not should_send(direction, last_state):
                print("Duplicate signal within cooldown period, not sent.")

