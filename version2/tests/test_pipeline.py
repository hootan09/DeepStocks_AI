import sys, pandas as pd
sys.path.append("src")
from src.features import add_indicators
def test_no_nan_after_indicators():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=100),
        "Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 1e6
    })
    df = add_indicators(df, type("cfg", (), {"rsi_window":14}))
    assert not df.isna().any().any()