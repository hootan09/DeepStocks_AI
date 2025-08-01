import yfinance as yf, finnhub, pandas as pd, numpy as np, os, random, datetime
from loguru import logger
from pathlib import Path

config_path = Path.cwd().parent
CACHE_DIR = config_path / ".cache" / "price_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """Return deterministic cache file name."""
    # key = f"{ticker}_{start}_{end}.parquet"
    key = f"{ticker}_{start}_{end}.csv"
    # Replace any '/' in dates to avoid sub-dirs
    return CACHE_DIR / key.replace("/", "-")


def _load_from_cache(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Load DataFrame from disk if it exists, else None."""
    path = _cache_path(ticker, start, end)
    if path.exists():
        logger.info(f"Loading cached data from {path}")
        # return pd.read_parquet(path)
        return pd.read_csv(path)
    return None


def _save_to_cache(df: pd.DataFrame, ticker: str, start: str, end: str) -> None:
    """Persist DataFrame to disk."""
    path = _cache_path(ticker, start, end)
    # df.to_parquet(path, index=False)
    df.to_csv(path, index=False)
    logger.info(f"Saved data to cache {path}")


def download_price_news(cfg):
    ticker = cfg.data.ticker
    start = cfg.data.start
    end = cfg.data.end or datetime.datetime.today().strftime("%Y-%m-%d")

    # -----------------------------------------------------------
    # 1. Try cache first
    # -----------------------------------------------------------
    if not getattr(cfg.data, "force_refresh", False):
        cached = _load_from_cache(ticker, start, end)
        if cached is not None:
            return cached

    # -----------------------------------------------------------
    # 2. Download (or mock) data
    # -----------------------------------------------------------
    price = yf.download(ticker, start=start, end=end).reset_index()
    price["ticker"] = ticker

    # Flatten the columns of the price DataFrame
    if isinstance(price.columns, pd.MultiIndex):
        price.columns = price.columns.get_level_values(0)


    logger.warning("Using synthetic news")
    fake = ["Apple announces product", "Market reacts", "Analyst upgrade"]
    news_df = pd.DataFrame(
        {
            "Date": price["Date"],
            "titles": [random.choice(fake) for _ in range(len(price))],
        }
    )

    # Convert the 'Date' columns to the same data type for merging
    price['Date'] = pd.to_datetime(price['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])

    df = price.merge(news_df, on="Date", how="left")
    df["titles"] = df["titles"].fillna("")

    # -----------------------------------------------------------
    # 3. Persist to cache
    # -----------------------------------------------------------
    _save_to_cache(df, ticker, start, end)
    return df


def walk_forward_split(df, train_pct, val_pct, purge):
    n = len(df)
    train_end = int(train_pct * n)
    val_end = int((train_pct + val_pct) * n)
    # purge = lookback to avoid look-ahead
    train = df.iloc[: train_end - purge]
    val = df.iloc[train_end : val_end - purge]
    test = df.iloc[val_end:]
    return train, val, test
