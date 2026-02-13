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
