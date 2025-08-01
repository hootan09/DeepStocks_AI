# src/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import torch
from pathlib import Path

from .model import StockNet
from .features import StockDataset   # re-use the dataset scaler helper
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="DeepStocks-API", version="1.0")

# ------------------------------------------------------------------
# 1) Lightweight runtime dependencies
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("../version2/models/best.pt")
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
BERT = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)

# ------------------------------------------------------------------
# 2) Load model & scaler once at startup
# ------------------------------------------------------------------
scaler = None   # will be replaced by a dummy scaler with correct n_features

@app.on_event("startup")
def load_model():
    global scaler, model
    # Load dummy dataset just to grab the scaler (cheap because no data)
    dummy_df = pd.DataFrame(0, index=range(35),
                            columns=["Open","High","Low","Close","Volume",
                                     "RSI","MACD","MACD_Signal","MACD_Hist",
                                     "BB_Upper","BB_Middle","BB_Lower","VWAP","label"])
    dummy_df["label"] = 0
    dummy_ds = StockDataset(dummy_df, torch.zeros(35, 768), lookback=30)
    scaler = dummy_ds.scaler

    # Load weights
    num_feats = scaler.n_features_in_
    model = StockNet(num_price_feat=num_feats).to(DEVICE)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH.resolve()}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

# ------------------------------------------------------------------
# 3) Request / Response schemas
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")
    price_df: list[dict] = Field(..., description="Last N rows of OHLCV+indicators")
    news: str = Field(..., example="Apple beats earnings estimates")

class PredictResponse(BaseModel):
    direction: str
    probability: float

# ------------------------------------------------------------------
# 4) Endpoint
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame(req.price_df)
        print(len(df))
        if len(df) < 30:
            raise ValueError("Need at least 30 rows")

        # Build tensor
        feats = df[["Close","High","Low","Open","Volume","RSI",
                    "MACD","MACD_Signal","MACD_Hist","BB_Upper",
                    "BB_Middle","BB_Lower","VWAP"]].values
        feats = torch.tensor(scaler.transform(feats[-30:]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Embed news
        tokens = TOKENIZER(req.news, return_tensors="pt",
                           truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            news_emb = BERT(**tokens).pooler_output

            out = model(feats, news_emb)
            prob = torch.softmax(out, dim=1)[0]
            direction = "UP" if prob[1] > 0.5 else "DOWN"

        return PredictResponse(direction=direction, probability=float(prob[1]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------------------------------------------------------
# 5) Health check
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}