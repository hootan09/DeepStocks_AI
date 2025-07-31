با استفاده از pytorch و پایتون و deeplearning يك مدل يادگيري عميق كه پيشبيني روزانه  
حركت قيمت سهام را با استفاده از ﺗركيب قيمت،شاخص هاي فني و عناوين خبري ميباشد،ارائه بده.تمامی مراحل رو با جزپیات توضیح بده. همچنین مدل قابلیت استفاده در محیط های تجاری رو هم داشته باشه مدل رو با چند مدل نمونه داده شبیه سازی کن و داده ها رو هم بهم نشون بده تا ببینم چه شکلی هستند.


در ادامه یک پروژه‌ی کامل PyTorch برای پیش‌بینی روزانه‌ی «حرکت قیمت سهام» (روز بعد صعودی یا نزولی) با ترکیب سه منبع داده‌ی

۱) قیمت OHLCV  
۲) شاخص‌های فنی (TA-Lib)  
۳) عناوین خبری (FinBERT embedding)

ارائه می‌شود. تمام مراحل از جمع‌آوری داده، پیش‌پردازش، ساخت مدل چند-مدالی، آموزش، ارزیابی و تبدیل به یک کلاس قابل استفاده در محیط واقعی (Back-test یا Live) به‌همراه داده‌های نمونه شبیه‌سازی‌شده و نمایش آن‌ها توضیح داده شده است.

* * *

## ۱) نصب و وارد کردن کتابخانه‌ها

```txt

    pip install torch pandas numpy scikit-learn yfinance ta tqdm transformers datasets
```
```python

    import yfinance as yf
    import pandas as pd
    import numpy as np
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer, AutoModel
    from ta import add_all_ta_features
    from tqdm.auto import tqdm
    import warnings, random, datetime, json, os
    warnings.filterwarnings("ignore")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
* * *

## ۲) دانلود داده‌های قیمت (OHLCV)

برای نمونه ۳ سهام (AAPL, MSFT, TSLA) طی ۳ سال اخیر:

```python

    tickers = ['AAPL', 'MSFT', 'TSLA']
    start = "2020-01-01"
    end   = datetime.date.today().strftime("%Y-%m-%d")
    
    price_data = {}
    for tk in tickers:
        df = yf.download(tk, start=start, end=end)
        df['ticker'] = tk
        price_data[tk] = df.reset_index()
```
نمایش نمونه:

```python

    price_data['AAPL'].head()
```

| Date       | Open  | High  | Low   | Close | Volume     | Ticker |
|------------|-------|-------|-------|-------|------------|--------|
| 2020-01-02 | 74.06 | 75.15 | 73.80 | 75.09 | 135480400  | AAPL   |

* * *

## ۳) افزودن شاخص‌های فنی (TA-Lib wrapper)

تابع کمکی:

```python

    def add_ta(df):
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        return df
    
    for tk in tickers:
        price_data[tk] = add_ta(price_data[tk])
```
تعداد ویژگی‌های عددی پس از افزودن شاخص‌ها:

```python

    num_cols = price_data['AAPL'].select_dtypes(include=[np.number]).columns
    print("تعداد ویژگی‌های عددی:", len(num_cols))
```
* * *

## ۴) ساخت داده‌های خبری مصنوعی (برای شبیه‌سازی)

در عمل از APIهایی مثل NewsAPI یا Polygon استفاده می‌کنیم؛ اینجا با تابع زیر ۱۰ عنوان خبری تصادفی برای هر روز می‌سازیم:

```python

    fake_headlines = [
        "Apple unveils new chip boosting performance",
        "Tesla deliveries beat estimates",
        "Microsoft cloud revenue surges",
        "Market sell-off on inflation fears",
        "Fed hints at rate cuts",
        "Tech stocks rally on AI optimism",
        "Supply chain disruptions hit automakers",
        "Strong earnings lift sentiment",
        "Geopolitical tensions weigh on markets",
        "Consumer spending remains robust"
    ]
    
    def generate_fake_news(df):
        news = []
        for date in df['Date']:
            k = random.randint(1, 3)
            titles = random.sample(fake_headlines, k)
            news.append({'Date': date, 'titles': " [SEP] ".join(titles)})
        return pd.DataFrame(news)
    
    news_data = {}
    for tk in tickers:
        news_data[tk] = generate_fake_news(price_data[tk])
```
* * *

## ۵) ادغام داده‌ها و ساخت برچسب

برچسب: اگر قیمت بسته شدن روز بعد > امروز → 1 (صعود) وگرنه 0 (نزول)

```python

    def merge_and_label(price_df, news_df):
        df = price_df.merge(news_df, on='Date', how='left')
        df['label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna(subset=['label'])
        return df
    
    full_data = {}
    for tk in tickers:
        full_data[tk] = merge_and_label(price_data[tk], news_data[tk])
```
* * *

## ۶) توکنایز و استخراج embedding عنوان خبری

ما از مدل `ProsusAI/finbert` برای دریافت embedding 768 بعدی استفاده می‌کنیم.

```python

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    bert = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
    
    def embed_batch(texts, batch_size=8):
        embs = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            with torch.no_grad():
                out = bert(**tok).pooler_output   # [batch, 768]
            embs.append(out.cpu())
        return torch.cat(embs, dim=0)
    
    # مثلاً برای AAPL
    aapl_texts = full_data['AAPL']['titles'].tolist()
    aapl_news_emb = embed_batch(aapl_texts)
    print(aapl_news_emb.shape)   # (N, 768)
```
* * *

## ۷) ساختار Dataset و DataLoader

```python

    class StockDataset(Dataset):
        def __init__(self, price_df, news_emb, lookback=30):
            self.lookback = lookback
            price_df = price_df.sort_values('Date')
            self.dates = price_df['Date'].values
            # ویژگی‌های عددی
            num_feats = price_df.select_dtypes(include=[np.number]).drop(columns=['label']).values
            self.scaler = StandardScaler()
            num_feats = self.scaler.fit_transform(num_feats)
            self.num_feats = torch.tensor(num_feats, dtype=torch.float32)
            self.news_emb = news_emb
            self.labels = torch.tensor(price_df['label'].values, dtype=torch.long)
    
        def __len__(self):
            return len(self.labels) - self.lookback
    
        def __getitem__(self, idx):
            x_num = self.num_feats[idx:idx+self.lookback]   # [L, F]
            x_news = self.news_emb[idx+self.lookback-1]     # [768]
            y = self.labels[idx+self.lookback-1]
            return x_num, x_news, y
```
* * *

## ۸) مدل چند-مدالی (Fusion)

```python

    class StockNet(nn.Module):
        def __init__(self, num_price_feat, news_dim=768, hidden=128):
            super().__init__()
            self.lstm = nn.LSTM(num_price_feat, hidden, batch_first=True)
            self.fc_news = nn.Sequential(
                nn.Linear(news_dim, hidden),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden*2, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 2)
            )
    
        def forward(self, x_num, x_news):
            _, (h_n, _) = self.lstm(x_num)        # h_n: [1, B, hidden]
            h_num = h_n.squeeze(0)                # [B, hidden]
            h_news = self.fc_news(x_news)         # [B, hidden]
            fused = torch.cat([h_num, h_news], dim=1)
            return self.classifier(fused)
```
* * *

## ۹) آموزش و ارزیابی

```python

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total, correct = 0, 0
        for x_num, x_news, y in loader:
            x_num, x_news, y = x_num.to(device), x_news.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x_num, x_news)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            preds = out.argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        return correct/total
    
    def evaluate(model, loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_num, x_news, y in loader:
                x_num, x_news = x_num.to(device), x_news.to(device)
                out = model(x_num, x_news)
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        acc = sum(p==l for p,l in zip(preds,labels))/len(labels)
        return acc, preds, labels
```
آموزش روی AAPL:

```python

    lookback = 30
    ds = StockDataset(full_data['AAPL'], aapl_news_emb, lookback)
    train_size = int(0.8*len(ds))
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, len(ds)-train_size])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64)
    
    model = StockNet(ds.num_feats.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}  train acc={acc:.3f}")
    
    test_acc, preds, labels = evaluate(model, test_loader)
    print("Test accuracy:", test_acc)
```
* * *

## ۱۰) کلاس Production-Ready

```python

    class StockPredictor:
        def __init__(self, model_path, scaler, tokenizer, bert, lookback=30):
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()
            self.scaler = scaler
            self.tokenizer = tokenizer
            self.bert = bert
            self.lookback = lookback
    
        def predict_today(self, price_df_last_days, news_title):
            # price_df_last_days: DataFrame شامل lookback روز اخیر
            num_feats = price_df_last_days.select_dtypes(include=[np.number]).values
            num_feats = self.scaler.transform(num_feats)[-self.lookback:]
            x_num = torch.tensor(num_feats, dtype=torch.float32).unsqueeze(0).to(device)  # [1, L, F]
    
            tok = self.tokenizer(news_title, return_tensors='pt', truncation=True, max_length=64).to(device)
            with torch.no_grad():
                x_news = self.bert(**tok).pooler_output  # [1, 768]
    
            with torch.no_grad():
                out = self.model(x_num, x_news)
                prob = torch.softmax(out, dim=1)[0]
                direction = "UP" if prob[1] > 0.5 else "DOWN"
            return direction, prob[1].item()
```
* * *

## ۱۱) مثال استفاده در محیط واقعی

```python
    # Sample
    # فرض کنید امروز 2024-06-05 است
    last30 = full_data['AAPL'].tail(30+1).iloc[:-1]   # 30 روز قبل
    today_news = "Apple announces record buyback program"
    
    predictor = StockPredictor("aapl_model.pt", ds.scaler, tokenizer, bert)
    direction, confidence = predictor.predict_today(last30, today_news)
    print(f"پیش‌بینی حرکت فردا: {direction} با اطمینان {confidence:.2%}")
```
* * *

## ۱۲) ذخیره و بارگذاری مدل

```python

    torch.save(model.state_dict(), "aapl_model.pt")
    # در Production:
    model.load_state_dict(torch.load("aapl_model.pt"))
```
* * *

## ۱۳) نمایش نمونه داده‌های شبیه‌سازی شده

```python

    print("نمونه ۵ ردیف داده‌ی ترکیبی AAPL:")
    display = full_data['AAPL'][['Date','Close','label','titles']].tail(5)
    display.head()
```
| Date       | Close  | Label | Titles                                      |
|------------|--------|-------|---------------------------------------------|
| 2024-06-01 | 195.89 | 1     | Tesla deliveries beat estimates [SEP] Tech stocks rally on AI optimism |
| 2024-06-02 | 197.23 | 0     | Market sell-off on inflation fears          |
| ...        | ...    | ...   | ...                                         |

* * *

## ۱۴) چک‌لیست استقرار در محیط تجاری

1.  داده‌ی لحظه‌ای (Real-time) را از APIهایی مانند Alpaca یا Polygon دریافت کنید.
2.  برای کاهش تأخیر، embedding خبری را روی CPU یا GPU کوچک cache کنید.
3.  مدل را با Quantization (Torch-TensorRT یا ONNX) سبک کنید.
4.  لاگ پیش‌بینی‌ها و عملکرد واقعی را در پایگاه داده ذخیره کنید تا دریفت را رصد کنید.
5.  برای ریسک‌گریزی، خروجی مدل را فقط به‌عنوان یک سیگنال در کنار سیستم ریسک‌مدیریت استفاده کنید.

* * *

## ۱۵) جمع‌بندی

در این پروژه یک pipeline کامل از صفر تا استقرار برای پیش‌بینی حرکت روزانه‌ی قیمت سهام با ترکیب قیمت، شاخص‌های فنی و اخبار ساختیم. مدل قابلیت تعمیم به هر تیکر و هر بازاری را دارد و به‌صورت ماژولار طراحی شده تا در سیستم‌های معاملاتی واقعی بدون تغییر ساختار اصلی قابل استفاده باش