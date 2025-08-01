import yfinance as yf, finnhub, pandas as pd, numpy as np, os, random, datetime
from loguru import logger

def download_price_news(cfg):
    ticker = cfg.data.ticker
    start  = cfg.data.start
    end    = cfg.data.end or datetime.datetime.today().strftime("%Y-%m-%d")

    price = yf.download(ticker, start=start, end=end).reset_index()
    price["ticker"] = ticker

    # --- NEWS ------------------------------------------------------
    if cfg.data.news_api == "finnhub" and cfg.data.finnhub_key:
        client = finnhub.Client(api_key=cfg.data.finnhub_key)
        news = []
        for d in pd.date_range(start, end, freq="D"):
            r = client.company_news(ticker, _from=d.strftime("%Y-%m-%d"), to=d.strftime("%Y-%m-%d"))
            titles = [x["headline"] for x in r]
            news.append({"Date": d, "titles": " [SEP] ".join(titles)})
        news_df = pd.DataFrame(news)
    else:
        logger.warning("Using synthetic news")
        fake = ["Apple announces product", "Market reacts", "Analyst upgrade"]
        news_df = pd.DataFrame({
            "Date": price["Date"],
            "titles": [random.choice(fake) for _ in range(len(price))]
        })

    df = price.merge(news_df, on="Date", how="left")
    df["titles"] = df["titles"].fillna("")
    return df

def walk_forward_split(df, train_pct, val_pct, purge):
    n = len(df)
    train_end = int(train_pct * n)
    val_end   = int((train_pct + val_pct) * n)
    # purge = lookback to avoid look-ahead
    train = df.iloc[:train_end - purge]
    val   = df.iloc[train_end:val_end - purge]
    test  = df.iloc[val_end:]
    return train, val, test