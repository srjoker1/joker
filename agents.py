"""
4 agent های تحلیل بازار طلا و نقره
هر ایجنت یک امتیاز بین -100 (شدیدا نزولی) تا +100 (شدیدا صعودی) برمی‌گردونه
به همراه دلیل تصمیمش (reason)
"""

import requests
import datetime
import numpy as np

TWELVEDATA_BASE = "https://api.twelvedata.com"


def get_time_series(symbol, interval="1h", outputsize=200, api_key=None):
    """قیمت‌های تاریخی رو از Twelve Data می‌گیره (فقط close)"""
    return get_ohlc(symbol, interval, outputsize, api_key)["close"]


def get_ohlc(symbol, interval="1h", outputsize=200, api_key=None):
    """قیمت‌های تاریخی کامل (open/high/low/close) رو می‌گیره"""
    url = f"{TWELVEDATA_BASE}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"خطا در گرفتن داده {symbol}: {data}")
    values = list(reversed(data["values"]))
    return {
        "close": np.array([float(v["close"]) for v in values]),
        "high": np.array([float(v["high"]) for v in values]),
        "low": np.array([float(v["low"]) for v in values]),
    }


def atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    trs = np.array(trs)
    if len(trs) < period:
        return float(np.mean(trs)) if len(trs) else 0.0
    return float(np.mean(trs[-period:]))


def _rsi_from_rs(up, down):
    if down == 0:
        return 100.0 if up > 0 else 50.0
    rs = up / down
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(closes, period=14):
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rsi_vals = np.zeros_like(closes)
    rsi_vals[:period] = _rsi_from_rs(up, down)
    for i in range(period, len(closes) - 1):
        delta = deltas[i]
        upval = delta if delta > 0 else 0
        downval = -delta if delta < 0 else 0
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rsi_vals[i + 1] = _rsi_from_rs(up, down)
    return rsi_vals


def ema(values, period):
    values = np.array(values, dtype=float)
    alpha = 2 / (period + 1)
    result = np.zeros_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def macd(closes, fast=12, slow=26, signal=9):
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line[-1], signal_line[-1], hist[-1]


class FundamentalAgent:
    name = "فاندامنتال (دلار)"

    def analyze(self, api_key):
        try:
            dxy = get_time_series("DXY", interval="1day", outputsize=30, api_key=api_key)
        except Exception:
            eurusd = get_time_series("EUR/USD", interval="1day", outputsize=30, api_key=api_key)
            dxy = 1 / eurusd

        change_5d = (dxy[-1] - dxy[-6]) / dxy[-6] * 100
        change_20d = (dxy[-1] - dxy[-21]) / dxy[-21] * 100 if len(dxy) > 21 else change_5d

        score = -(change_5d * 6 + change_20d * 2)
        score = max(-100, min(100, score))

        direction = "دلار در حال تقویته" if change_5d > 0 else "دلار در حال تضعیفه"
        reason = (
            f"{direction} (تغییر ۵ روزه: {change_5d:.2f}%، ۲۰ روزه: {change_20d:.2f}%) "
            f"→ اثر {'منفی' if change_5d > 0 else 'مثبت'} روی طلا/نقره"
        )
        return round(score, 1), reason


class SilverIndustrialAgent:
    name = "نقره صنعتی / نسبت طلا-نقره"

    def analyze(self, api_key):
        gold = get_time_series("XAU/USD", interval="1day", outputsize=60, api_key=api_key)
        silver = get_time_series("XAG/USD", interval="1day", outputsize=60, api_key=api_key)

        ratio = gold[-1] / silver[-1]
        ratio_avg = np.mean(gold[-30:] / silver[-30:])

        deviation = (ratio - ratio_avg) / ratio_avg * 100
        score = deviation * 8
        score = max(-100, min(100, score))

        silver_vol = np.std(np.diff(silver[-20:]) / silver[-21:-1]) * 100
        reason = (
            f"نسبت طلا/نقره فعلی: {ratio:.1f} (میانگین ۳۰ روزه: {ratio_avg:.1f}, "
            f"انحراف: {deviation:.2f}%) | نوسان روزانه نقره اخیر: {silver_vol:.2f}%"
        )
        return round(score, 1), reason


class TechnicalAgent:
    name = "تحلیل تکنیکال"

    def _score_symbol(self, ohlc):
        closes, highs, lows = ohlc["close"], ohlc["high"], ohlc["low"]
        r = rsi(closes)[-1]
        macd_line, signal_line, hist = macd(closes)
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else ma20
        price = closes[-1]
        atr_val = atr(highs, lows, closes)

        score = 0
        notes = []

        if r < 30:
            score += 35
            notes.append(f"RSI={r:.0f} اشباع فروش")
        elif r > 70:
            score -= 35
            notes.append(f"RSI={r:.0f} اشباع خرید")
        else:
            notes.append(f"RSI={r:.0f} خنثی")

        if hist > 0:
            score += 25
            notes.append("MACD صعودی")
        else:
            score -= 25
            notes.append("MACD نزولی")

        if price > ma20 > ma50:
            score += 20
            notes.append("روند صعودی (قیمت بالای MA20 و MA50)")
        elif price < ma20 < ma50:
            score -= 20
            notes.append("روند نزولی (قیمت زیر MA20 و MA50)")

        score = max(-100, min(100, score))
        return score, ", ".join(notes), price, atr_val

    def analyze(self, api_key):
        gold_ohlc = get_ohlc("XAU/USD", interval="4h", outputsize=200, api_key=api_key)
        silver_ohlc = get_ohlc("XAG/USD", interval="4h", outputsize=200, api_key=api_key)

        gold_score, gold_notes, gold_price, gold_atr = self._score_symbol(gold_ohlc)
        silver_score, silver_notes, silver_price, silver_atr = self._score_symbol(silver_ohlc)

        avg_score = (gold_score + silver_score) / 2
        reason = f"طلا: {gold_notes} | نقره: {silver_notes}"

        levels = {
            "XAU/USD": {"score": gold_score, "price": gold_price, "atr": gold_atr},
            "XAG/USD": {"score": silver_score, "price": silver_price, "atr": silver_atr},
        }
        return round(avg_score, 1), reason, levels


class CalendarAgent:
    name = "تقویم اقتصادی (ریسک رویداد)"

    FOMC_DATES_2026 = [
        ("2026-01-27", "2026-01-28"),
        ("2026-03-17", "2026-03-18"),
        ("2026-04-28", "2026-04-29"),
        ("2026-06-16", "2026-06-17"),
        ("2026-07-28", "2026-07-29"),
        ("2026-09-15", "2026-09-16"),
        ("2026-10-27", "2026-10-28"),
        ("2026-12-08", "2026-12-09"),
    ]

    def _first_friday(self, year, month):
        d = datetime.date(year, month, 1)
        while d.weekday() != 4:
            d += datetime.timedelta(days=1)
        return d

    def analyze(self, api_key=None):
        today = datetime.date.today()
        risk_notes = []
        risk_score = 0

        for start, end in self.FOMC_DATES_2026:
            start_d = datetime.date.fromisoformat(start)
            end_d = datetime.date.fromisoformat(end)
            if start_d - datetime.timedelta(days=2) <= today <= end_d + datetime.timedelta(days=1):
                risk_notes.append(f"نزدیک جلسه FOMC ({start} تا {end}) - نوسان بالا محتمله")
                risk_score += 40

        nfp_day = self._first_friday(today.year, today.month)
        if abs((today - nfp_day).days) <= 1:
            risk_notes.append(f"نزدیک گزارش اشتغال آمریکا NFP ({nfp_day.isoformat()})")
            risk_score += 30

        if not risk_notes:
            risk_notes.append("رویداد کلان مهم و شناخته‌شده‌ای در این بازه نیست")

        return risk_score, " | ".join(risk_notes)
