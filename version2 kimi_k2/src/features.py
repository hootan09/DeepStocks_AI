from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch, numpy as np

def add_indicators(df, cfg):
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["RSI"] = RSIIndicator(df["Close"], cfg.rsi_window).rsi()
    macd = MACD(df["Close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_sign)
    df["MACD"] = macd.macd(); df["MACD_Signal"] = macd.macd_signal(); df["MACD_Hist"] = macd.macd_diff()
    bb = BollingerBands(df["Close"], cfg.bb_window, cfg.bb_std)
    df["BB_Upper"]=bb.bollinger_hband(); df["BB_Middle"]=bb.bollinger_mavg(); df["BB_Lower"]=bb.bollinger_lband()
    df["VWAP"] = VolumeWeightedAveragePrice(df["High"],df["Low"],df["Close"],df["Volume"],cfg.vwap_window).volume_weighted_average_price()
    df = df.dropna().reset_index(drop=True)
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.iloc[:-1]   # drop last row (no label)

class StockDataset(Dataset):
    def __init__(self, price_df, news_embs, lookback):
        self.lookback = lookback
        self.price = price_df.select_dtypes(include=[np.number]).drop(columns=["label"])
        self.news  = news_embs
        self.label = torch.tensor(price_df["label"].values, dtype=torch.long)
        self.scaler = StandardScaler().fit(self.price.values)
        self.price  = torch.tensor(self.scaler.transform(self.price.values), dtype=torch.float32)
    def __len__(self): return len(self.label) - self.lookback
    def __getitem__(self, idx):
        return (self.price[idx:idx+self.lookback],
                self.news[idx+self.lookback-1],
                self.label[idx+self.lookback-1])

def build_dataset(train_df, val_df, test_df, lookback):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    bert = AutoModel.from_pretrained("ProsusAI/finbert")
    def embed(texts):
        tokens = tokenizer(texts.tolist(), padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
        with torch.no_grad():
            return bert(**tokens).pooler_output
    train_emb = embed(train_df["titles"])
    val_emb   = embed(val_df["titles"])
    test_emb  = embed(test_df["titles"])
    return (StockDataset(train_df, train_emb, lookback),
            StockDataset(val_df,   val_emb,   lookback),
            StockDataset(test_df,  test_emb,  lookback),
            train_ds.scaler)   # return only training scaler