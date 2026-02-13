# AI Trading Bot for XAUUSD (MetaTrader 5 + Python)

این پروژه یک چارچوب کامل برای:
1. دریافت دیتای تاریخی XAUUSD از MetaTrader 5
2. ساخت ویژگی‌های تکنیکال
3. آموزش مدل ML روی 6 ماه / 1 سال داده
4. تولید سیگنال روی آخرین کندل‌ها (مثلاً 100 کندل اخیر)

> ⚠️ این پروژه صرفاً آموزشی است و سیگنال مالی/سرمایه‌گذاری قطعی نیست.

## ساختار پروژه

- `ai_trader/config.py` تنظیمات مرکزی
- `ai_trader/data.py` اتصال و دریافت داده از MT5
- `ai_trader/features.py` ساخت Feature و Label
- `ai_trader/train.py` آموزش مدل و ذخیره artifact
- `ai_trader/signal.py` تولید سیگنال زنده از آخرین داده
- `tests/test_features.py` تست واحد feature engineering

## نصب

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## پیش‌نیاز متاتریدر5

- متاتریدر5 روی سیستم نصب باشد.
- حساب (دمو یا ریل) لاگین باشد.
- نماد `XAUUSD` در Market Watch فعال باشد.


## الان روی چه تایم‌فریمی کار می‌کند؟

در وضعیت فعلی، پیش‌فرض پروژه روی `H1` است (قابل تغییر با `--timeframe`).

- پیش‌فرض در تنظیمات: `H1`
- نمونه اجرا برای `M30`: `python -m ai_trader.train --timeframe M30 --months 6`

## چه تایم‌فریمی بهتر است؟

برای XAUUSD معمولاً:
- `M15`: سیگنال بیشتر، نویز بیشتر، هزینه معامله بالاتر
- `M30`: تعادل مناسب بین سرعت و نویز
- `H1`: پایدارتر برای شروع (پیشنهاد اولیه)
- `H4`: سیگنال کمتر ولی با اطمینان نسبی بیشتر

پیشنهاد عملی: از `H1` شروع کن، بعد `M30` و `H4` را با داده یکسان مقایسه کن.

برای همین یک ابزار مقایسه اضافه شده:

```bash
python -m ai_trader.compare_timeframes --months 12 --timeframes M15 M30 H1 H4
```

این اسکریپت برای هر تایم‌فریم، ROC-AUC و نرخ کلاس مثبت را چاپ می‌کند و بهترین تایم‌فریم را رتبه‌بندی می‌کند.

## آموزش مدل

### یک سال اخیر
```bash
python -m ai_trader.train --months 12 --timeframe H1
```

### شش ماه اخیر
```bash
python -m ai_trader.train --months 6 --timeframe M30
```

خروجی:
- `artifacts/model.joblib` مدل
- `artifacts/metrics.json` متریک‌های آموزش/تست

## تولید سیگنال

```bash
python -m ai_trader.signal --lookback 100 --threshold 0.58
```

خروجی نمونه:
```json
{
  "timestamp": "2026-02-13 12:00:00",
  "symbol": "XAUUSD",
  "signal": "BUY",
  "prob_up": 0.64,
  "threshold": 0.58
}
```

## منطق مدل

- برچسب (Label):
  - اگر بازده `N` کندل آینده > `target_return` → کلاس 1 (UP)
  - در غیر این صورت کلاس 0 (NOT_UP)
- ویژگی‌ها:
  - بازده‌های لگ‌دار
  - ATR نرمال‌شده
  - میانگین/انحراف معیار rolling
  - فاصله قیمت از EMA
- مدل:
  - `HistGradientBoostingClassifier` (سریع، مناسب Tabular)

## پیشنهاد بهتر از «فقط نگاه به 100 کندل»

این پروژه ترکیب زیر را پیاده کرده:
- تصمیم بر اساس featureهای آماری + تکنیکال (نه صرفاً Raw Candle)
- خروجی احتمالی (`prob_up`) + آستانه قابل تنظیم (`threshold`)
- جلوگیری از سیگنال‌های نویزی با افزایش threshold (مثلاً 0.60)

برای ارتقای بعدی:
- Walk-forward validation
- رژیم بازار (trend/range classifier)
- مدیریت ریسک پویا با ATR
- هزینه معامله/اسلیپیج در بک‌تست

## هشدار ریسک

هر مدل ML روی بازار ممکن است overfit شود. قبل از استفاده واقعی:
- بک‌تست چندساله
- فوروارد تست روی دمو
- محدودیت ریسک سخت‌گیرانه (حداکثر افت سرمایه)


## اجرای دائم بات روی بسته‌شدن کندل (Auto Trading)

اگر می‌خواهی بات قطع نشود و روی **همان تایم‌فریم مدل** در لحظه بسته‌شدن کندل سیگنال بدهد و خودش پوزیشن باز کند، از این دستور استفاده کن:

```bash
python -m ai_trader.live_bot \
  --model-path artifacts/model.joblib \
  --poll-seconds 5 \
  --buy-threshold 0.58 \
  --sell-threshold 0.42 \
  --lot 0.01 \
  --sl-points 400 \
  --tp-points 800 \
  --allow-trading
```

نکات مهم:
- این اسکریپت دائمی اجرا می‌شود (`while True`) و هر چند ثانیه یک‌بار بررسی می‌کند.
- فقط وقتی کندل جدید بسته شد تصمیم می‌گیرد (پس روی هر تیک ترید نمی‌کند).
- اگر روی نماد پوزیشن باز داشته باشی، پوزیشن جدید باز نمی‌کند.
- برای تایم‌فریم 15 دقیقه، مدل باید با `--timeframe M15` آموزش داده شده باشد.

نمونه آموزش مدل 15 دقیقه:

```bash
python -m ai_trader.train --months 12 --timeframe M15
```

سپس همان مدل را به `live_bot` بده.
