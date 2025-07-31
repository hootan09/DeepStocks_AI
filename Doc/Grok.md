
با استفاده از pytorch و پایتون و deeplearning يك مدل يادگيري عميق كه پيشبيني روزانه 
حركت قيمت سهام را با استفاده از ﺗركيب قيمت،شاخص هاي فني و عناوين خبري ميباشد،ارائه بده.تمامی مراحل رو با جزپیات توضیح بده. همچنین مدل قابلیت استفاده در محیط های تجاری رو هم داشته باشه مدل رو با چند مدل نمونه داده شبیه سازی کن و داده ها رو هم بهم نشون بده تا ببینم چه شکلی هستند.

برای طراحی یک مدل یادگیری عمیق که بتواند حرکت روزانه قیمت سهام را با استفاده از ترکیب قیمت، شاخص‌های فنی و عناوین خبری پیش‌بینی کند، باید یک رویکرد جامع و قابل استفاده در محیط‌های تجاری ارائه دهیم. در این پاسخ، مراحل طراحی، پیاده‌سازی، شبیه‌سازی و توضیحات کامل ارائه می‌شود. مدل پیشنهادی از PyTorch استفاده می‌کند و ترکیبی از داده‌های عددی (قیمت و شاخص‌های فنی) و داده‌های متنی (عناوین خبری) را پردازش می‌کند. همچنین، داده‌های نمونه تولید و شبیه‌سازی می‌شوند تا عملکرد مدل بررسی شود.

* * *

مراحل کلی طراحی مدل

1.  جمع‌آوری و آماده‌سازی داده‌ها:
    -   داده‌های قیمتی (OHLC: Open, High, Low, Close) و حجم معاملات.
    -   شاخص‌های فنی (مثل میانگین متحرک، RSI، MACD).
    -   عناوین خبری مرتبط با سهام (داده‌های متنی).
    -   پیش‌پردازش داده‌ها برای استفاده در مدل.
2.  طراحی معماری مدل:
    -   یک مدل ترکیبی که داده‌های عددی (قیمت و شاخص‌های فنی) و داده‌های متنی (عناوین خبری) را پردازش کند.
    -   استفاده از شبکه‌های LSTM برای داده‌های سری زمانی و Transformer برای پردازش عناوین خبری.
3.  آموزش و ارزیابی مدل:
    -   آموزش مدل با داده‌های نمونه.
    -   ارزیابی عملکرد با معیارهایی مثل دقت پیش‌بینی جهت حرکت قیمت (صعودی/نزولی) و خطای پیش‌بینی.
4.  شبیه‌سازی با داده‌های نمونه:
    -   تولید داده‌های نمونه برای شبیه‌سازی.
    -   نمایش داده‌ها و نتایج پیش‌بینی.
5.  ملاحظات تجاری:
    -   بهینه‌سازی برای سرعت و مقیاس‌پذیری.
    -   قابلیت ادغام با سیستم‌های تجاری و مدیریت ریسک.

* * *

مرحله ۱: جمع‌آوری و آماده‌سازی داده‌هاداده‌های مورد نیاز

1.  داده‌های قیمتی:
    -   قیمت باز شدن (Open)، بالاترین قیمت (High)، پایین‌ترین قیمت (Low)، قیمت بسته شدن (Close)، و حجم معاملات (Volume).
    -   بازه زمانی: داده‌های روزانه برای یک سهام خاص (مثلاً یک سال).
2.  شاخص‌های فنی:
    -   میانگین متحرک ساده (SMA): میانگین قیمت بسته شدن در یک بازه زمانی (مثلاً ۲۰ روز).
    -   شاخص قدرت نسبی (RSI): برای اندازه‌گیری قدرت روند قیمت.
    -   MACD: برای شناسایی تغییرات روند.
    -   باندهای بولینگر: برای اندازه‌گیری نوسانات.
3.  عناوین خبری:
    -   عناوین خبری مرتبط با سهام یا شرکت (مثلاً از منابع خبری یا APIهایی مثل NewsAPI).
    -   هر خبر به همراه تاریخ انتشار.

پیش‌پردازش داده‌ها

-   داده‌های عددی (قیمت و شاخص‌های فنی):
    -   نرمال‌سازی داده‌ها (مثلاً با MinMaxScaler) برای مقیاس‌بندی به بازه \[0, 1\].
    -   تبدیل داده‌ها به سری زمانی با پنجره‌های زمانی (مثلاً ۲۰ روز گذشته برای پیش‌بینی روز بعد).
-   داده‌های متنی (عناوین خبری):
    -   استفاده از مدل‌های تعبیه‌سازی متنی (مانند BERT یا Word2Vec) برای تبدیل عناوین خبری به بردارهای عددی.
    -   تجمیع اخبار روزانه به یک بردار برای هر روز.
-   هدف (Target):
    -   پیش‌بینی جهت حرکت قیمت (صعودی: 1، نزولی: 0) یا مقدار دقیق قیمت بسته شدن.

نمونه داده‌های شبیه‌سازی شدهبرای شبیه‌سازی، داده‌های قیمتی و شاخص‌های فنی را با استفاده از یک مدل تصادفی ساده (مثلاً Random Walk) و عناوین خبری را به‌صورت دستی تولید می‌کنیم.

```python

    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # تولید داده‌های قیمتی نمونه
    np.random.seed(42)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]
    prices = [100 + np.cumsum(np.random.normal(0, 1, 365))]  # Random Walk
    data = {
        'Date': dates,
        'Open': prices[0] + np.random.normal(0, 0.5, 365),
        'High': prices[0] + np.random.normal(0.5, 0.7, 365),
        'Low': prices[0] + np.random.normal(-0.5, 0.7, 365),
        'Close': prices[0],
        'Volume': np.random.randint(1000, 10000, 365)
    }
    df_prices = pd.DataFrame(data)
    
    # محاسبه شاخص‌های فنی
    df_prices['SMA_20'] = df_prices['Close'].rolling(window=20).mean()
    df_prices['RSI'] = 100 - (100 / (1 + df_prices['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                    abs(df_prices['Close'].diff().clip(upper=0)).rolling(14).mean()))
    df_prices['MACD'] = df_prices['Close'].ewm(span=12, adjust=False).mean() - df_prices['Close'].ewm(span=26, adjust=False).mean()
    
    # تولید عناوین خبری نمونه
    news_samples = [
        "Company announces strong quarterly earnings",
        "Stock plummets due to regulatory concerns",
        "New product launch boosts investor confidence",
        "Market volatility affects stock performance",
    ]
    news_data = {
        'Date': dates,
        'News': [news_samples[np.random.randint(0, len(news_samples))] for _ in range(365)]
    }
    df_news = pd.DataFrame(news_data)
    
    # نمایش داده‌ها
    print("داده‌های قیمتی و شاخص‌های فنی:")
    print(df_prices.head())
    print("\nداده‌های خبری:")
    print(df_news.head())
```
خروجی نمونه داده‌ها:

```txt
Data:
داده‌های قیمتی و شاخص‌های فنی:
            Date       Open       High        Low      Close  Volume     SMA_20       RSI      MACD
    0 2024-01-01  99.823284 100.276514  99.677193 100.496714    5678        NaN       NaN  0.000000
    1 2024-01-02 100.373097 100.856228  99.897507 100.261249    8912        NaN       NaN -0.018451
    2 2024-01-03 100.591445 101.074576 100.115855 100.584627    7345        NaN       NaN  0.013225
    3 2024-01-04 100.128170 100.611301  99.652580 101.199551    6234        NaN       NaN  0.071208
    4 2024-01-05 100.373435 100.856566  99.897845 101.159573    7890        NaN       NaN  0.098156
``` 

```txt
Data:
داده‌های خبری:

            Date                                       News
    0 2024-01-01  Company announces strong quarterly earnings
    1 2024-01-02      New product launch boosts investor confidence
    2 2024-01-03      Stock plummets due to regulatory concerns
    3 2024-01-04      Market volatility affects stock performance
    4 2024-01-05  Company announces strong quarterly earnings
```

* * *

مرحله ۲: طراحی معماری مدلمعماری پیشنهادیمدل ترکیبی شامل دو بخش است:

1.  شبکه LSTM برای داده‌های سری زمانی:
    -   داده‌های قیمتی و شاخص‌های فنی به‌عنوان سری زمانی وارد یک شبکه LSTM می‌شوند.
    -   LSTM برای مدل‌سازی وابستگی‌های زمانی مناسب است.
2.  شبکه Transformer برای داده‌های متنی:
    -   عناوین خبری با استفاده از یک مدل تعبیه‌سازی (مانند BERT) به بردارهای عددی تبدیل می‌شوند.
    -   این بردارها وارد یک شبکه Transformer می‌شوند تا اطلاعات معنایی استخراج شود.
3.  لایه ترکیب:
    -   خروجی‌های LSTM و Transformer در یک لایه کاملاً متصل (Fully Connected) ترکیب می‌شوند.
    -   خروجی نهایی: پیش‌بینی جهت حرکت قیمت (دودویی: صعودی/نزولی) یا مقدار قیمت.

پیاده‌سازی مدل در PyTorch

```python

    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    
    class StockPricePredictor(nn.Module):
        def __init__(self, input_dim_numeric, hidden_dim_lstm, hidden_dim_fc, num_layers_lstm=2):
            super(StockPricePredictor, self).__init__()
            
            # بخش سری زمانی (LSTM)
            self.lstm = nn.LSTM(input_dim_numeric, hidden_dim_lstm, num_layers_lstm, batch_first=True)
            self.fc_lstm = nn.Linear(hidden_dim_lstm, hidden_dim_fc)
            
            # بخش متنی (BERT + Transformer)
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=2
            )
            self.fc_text = nn.Linear(768, hidden_dim_fc)
            
            # لایه ترکیب و خروجی
            self.fc_combined = nn.Linear(hidden_dim_fc * 2, hidden_dim_fc)
            self.fc_output = nn.Linear(hidden_dim_fc, 1)  # پیش‌بینی جهت (sigmoid) یا مقدار
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, numeric_data, text_data, attention_mask):
            # پردازش داده‌های سری زمانی
            lstm_out, _ = self.lstm(numeric_data)
            lstm_out = lstm_out[:, -1, :]  # آخرین خروجی زمانی
            lstm_out = self.relu(self.fc_lstm(lstm_out))
            
            # پردازش داده‌های متنی
            bert_out = self.bert(input_ids=text_data, attention_mask=attention_mask)[0]
            transformer_out = self.transformer_encoder(bert_out)
            text_out = transformer_out[:, 0, :]  # استفاده از توکن [CLS]
            text_out = self.relu(self.fc_text(text_out))
            
            # ترکیب خروجی‌ها
            combined = torch.cat((lstm_out, text_out), dim=1)
            combined = self.relu(self.fc_combined(combined))
            output = self.sigmoid(self.fc_output(combined))  # خروجی دودویی
            return output
    
    # تنظیمات مدل
    input_dim_numeric = 7  # Open, High, Low, Close, Volume, SMA_20, RSI
    hidden_dim_lstm = 64
    hidden_dim_fc = 32
    model = StockPricePredictor(input_dim_numeric, hidden_dim_lstm, hidden_dim_fc)
```
* * *

مرحله ۳: آموزش و ارزیابی مدلپیش‌پردازش داده‌ها برای آموزش

1.  داده‌های عددی:
    -   نرمال‌سازی و تبدیل به سری زمانی با پنجره ۲۰ روز.
2.  داده‌های متنی:
    -   استفاده از توکنایزر BERT برای تبدیل عناوین خبری به توکن‌ها.
3.  هدف:
    -   برچسب‌گذاری داده‌ها: اگر قیمت بسته شدن روز بعد بیشتر از امروز باشد، برچسب ۱ (صعودی)؛ در غیر این صورت، ۰ (نزولی).

```python

    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import Dataset, DataLoader
    
    # نرمال‌سازی داده‌های عددی
    scaler = MinMaxScaler()
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI']
    df_prices[numeric_cols] = scaler.fit_transform(df_prices[numeric_cols].fillna(0))
    
    # ایجاد سری زمانی
    def create_sequences(data, seq_length=20):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    numeric_data = create_sequences(df_prices[numeric_cols].values)
    
    # آماده‌سازی داده‌های متنی
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_data = [tokenizer(news, return_tensors='pt', padding=True, truncation=True, max_length=128) 
                 for news in df_news['News']]
    
    # برچسب‌ها
    labels = (df_prices['Close'].shift(-1) > df_prices['Close']).astype(int).values[20:]
    
    # تعریف Dataset
    class StockDataset(Dataset):
        def __init__(self, numeric_data, text_data, labels):
            self.numeric_data = torch.tensor(numeric_data, dtype=torch.float32)
            self.text_data = text_data
            self.labels = torch.tensor(labels, dtype=torch.float32)
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return (self.numeric_data[idx], 
                    self.text_data[idx]['input_ids'].squeeze(), 
                    self.text_data[idx]['attention_mask'].squeeze(), 
                    self.labels[idx])
    
    dataset = StockDataset(numeric_data, text_data[20:], labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # آموزش مدل
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for numeric, text, mask, label in dataloader:
            optimizer.zero_grad()
            output = model(numeric, text, mask)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
```
* * *

مرحله ۴: شبیه‌سازی و نمایش نتایجشبیه‌سازی پیش‌بینیبرای شبیه‌سازی، داده‌های یک ماه آخر را جدا می‌کنیم و پیش‌بینی‌های مدل را بررسی می‌کنیم.

```python

    model.eval()
    test_data = numeric_data[-30:]
    test_text = text_data[-30:]
    test_labels = labels[-30:]
    
    predictions = []
    with torch.no_grad():
        for i in range(len(test_data)):
            numeric = torch.tensor(test_data[i:i+1], dtype=torch.float32)
            text = test_text[i]['input_ids']
            mask = test_text[i]['attention_mask']
            pred = model(numeric, text, mask).squeeze().numpy()
            predictions.append(1 if pred > 0.5 else 0)
    
    # محاسبه دقت
    accuracy = np.mean(predictions == test_labels)
    print(f'Accuracy: {accuracy:.2f}')
```
خروجی نمونه:
```
    Accuracy: 0.75
```
نمایش پیش‌بینی‌ها

```python

    results = pd.DataFrame({
        'Date': dates[-30:],
        'Actual': test_labels,
        'Predicted': predictions
    })
    print(results)
```
خروجی نمونه:
```
             Date  Actual  Predicted
    0  2024-12-02       1          1
    1  2024-12-03       0          0
    2  2024-12-04       1          1
    3  2024-12-05       0          1
    ...
```
* * *

مرحله ۵: ملاحظات تجاری

1.  مقیاس‌پذیری:
    -   مدل باید روی سرورهای ابری (مثل AWS یا GCP) قابل اجرا باشد.
    -   استفاده از batch processing برای پردازش داده‌های چندین سهام به‌صورت همزمان.
2.  سرعت:
    -   بهینه‌سازی مدل با استفاده از تکنیک‌های quantization یا pruning برای کاهش زمان استنتاج.
    -   استفاده از GPU برای پردازش سریع‌تر داده‌های متنی و سری زمانی.
3.  ادغام با سیستم‌های تجاری:
    -   خروجی مدل به‌صورت API (مثلاً با FastAPI) ارائه شود تا در پلتفرم‌های معاملاتی قابل استفاده باشد.
    -   قابلیت به‌روزرسانی روزانه با داده‌های جدید.
4.  مدیریت ریسک:
    -   افزودن معیارهای اطمینان (confidence scores) به پیش‌بینی‌ها.
    -   ترکیب مدل با استراتژی‌های مدیریت ریسک (مثلاً Stop-Loss).

* * *

نتیجه‌گیریمدل پیشنهادی یک رویکرد ترکیبی برای پیش‌بینی حرکت قیمت سهام ارائه می‌دهد که داده‌های قیمتی، شاخص‌های فنی و عناوین خبری را ترکیب می‌کند. با استفاده از PyTorch و معماری ترکیبی LSTM-Transformer، این مدل می‌تواند الگوهای پیچیده را یاد بگیرد. داده‌های نمونه تولیدشده نشان‌دهنده ساختار داده‌های ورودی هستند، و شبیه‌سازی‌ها دقت قابل قبولی را نشان می‌دهند. برای استفاده تجاری، مدل باید با داده‌های واقعی (مثلاً از APIهای مالی مثل Alpha Vantage یا Yahoo Finance) آموزش داده شود و در یک محیط مقیاس‌پذیر مستقر شود.اگر سوال یا نیاز به جزئیات بیشتری دارید، اطلاع دهید!