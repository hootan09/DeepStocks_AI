import torch, yfinance as yf, datetime
class StockPredictor:
    def __init__(self, ckpt_path, scaler, lookback):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StockNet(scaler.n_features_in_).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        self.scaler = scaler
        self.lookback = lookback
    def predict_one(self, price_df, news):
        import pandas as pd
        feats = price_df[["Close","High","Low","Open","Volume","RSI","MACD","MACD_Signal","MACD_Hist","BB_Upper","BB_Middle","BB_Lower","VWAP"]].values
        feats = torch.tensor(self.scaler.transform(feats[-self.lookback:]), dtype=torch.float32).unsqueeze(0).to(self.device)
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        bert = AutoModel.from_pretrained("ProsusAI/finbert")
        toks = tok(news, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            news_emb = bert(**toks).pooler_output.to(self.device)
            out = self.model(feats, news_emb)
            prob = torch.softmax(out, dim=1)[0]
        return ("UP" if prob[1] > 0.5 else "DOWN", prob[1].item())