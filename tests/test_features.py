from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trader.features import build_dataset


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = pd.date_range("2025-01-01", periods=n, freq="H")
    base = np.linspace(2600, 2700, n)
    noise = np.random.default_rng(42).normal(0, 1.2, n)
    close = base + noise

    df = pd.DataFrame(
        {
            "timestamp": rng,
            "open": close + 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": 100,
        }
    )
    return df


def test_build_dataset_shapes():
    df = _sample_df(240)
    X, y, meta, cols = build_dataset(df, horizon=3, target_return=0.0005)

    assert len(cols) > 0
    assert len(X) == len(y) == len(meta)
    assert set(cols).issubset(set(X.columns))
    assert y.nunique() >= 1
