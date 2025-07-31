در اینجا یک راه‌حل کامل برای ساخت مدل یادگیری عمیق ترکیبی با استفاده از PyTorch برای پیش‌بینی روزانه حرکت قیمت سهام با ترکیب داده‌های قیمت، شاخص‌های تکنیکال و عناوین خبری ارائه می‌دهم. این راه‌حل شامل تمامی مراحل از جمع‌آوری داده تا آموزش مدل و شبیه‌سازی است.

* * *

### **مرحله ۱: جمع‌آوری داده‌ها**

#### ۱. داده‌های قیمت سهام:

-   **منبع**: Yahoo Finance، Alpha Vantage، یا API‌های مشابه.
-   **فیلدها**: `Open`, `High`, `Low`, `Close`, `Volume` (OHLCV).
-   **فرمت نمونه**:
    
```python
    

    {
      "Date": "2023-10-01",
      "Open": 150.2,
      "High": 152.5,
      "Low": 149.0,
      "Close": 151.8,
      "Volume": 1000000
    }
```
#### ۲. شاخص‌های تکنیکال:

-   **شاخص‌ها**: میانگین متحرک (SMA/EMA)، RSI، MACD، Bollinger Bands.
-   **محاسبه**: با کتابخانه `pandas` یا `ta`:
    
```python
    

    import ta
    df = ta.add_all_ta_features(df, colprefix="Close", fillna=True)
```
#### ۳. عناوین خبری:

-   **منبع**: NewsAPI، RSS Feed، یا وب‌اسکرپینگ.
-   **فرمت نمونه**:
    
```python
    

    {
      "Date": "2023-10-01",
      "News_Title": "شرکت X رکورد فروش جدیدی ثبت کرد"
    }
```
#### **توضیحات**:

-   داده‌ها باید بر اساس تاریخ هم‌زمان شوند (Time-Series Alignment).
-   برای شبیه‌سازی، داده‌های مصنوعی ایجاد می‌کنیم.

* * *

### **مرحله ۲: پیش‌پردازش داده‌ها**

#### ۱. داده‌های عددی (قیمت و شاخص‌ها):

-   **نرمال‌سازی**: استفاده از `MinMaxScaler` یا `StandardScaler`.
-   **تبدیل به توالی**: برای مدل‌های LSTM/GRU، داده‌ها را به شکل توالی (Sequence) درآورید:
    
```python
    

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["Close", "RSI", "MACD"]])
```
#### ۲. داده‌های متنی (عناوین خبری):

-   **توکنایزاسیون**: استفاده از `BERT Tokenizer` یا `TF-IDF`.
-   **Embedding**: استفاده از مدل‌های پیش‌آموزش‌دیده مثل `BERT` یا `GloVe`.
    
```python
    

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512)
```
#### **توضیحات**:

-   داده‌های متنی و عددی باید در مراحل جداگانه پردازش شوند و سپس ترکیب شوند.

* * *

### **مرحله ۳: طراحی معماری مدل**

#### **مدل ترکیبی (Hybrid Model)**:

-   **بخش ۱: پردازش داده‌های عددی**:
    
```python
    

    import torch.nn as nn
    class NumericalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
            self.fc = nn.Linear(64, 32)
```
-   **بخش ۲: پردازش داده‌های متنی**:
    
```python
    

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.fc = nn.Linear(768, 32)
```
-   **ترکیب دو خروجی**:
    
```python
    

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.numerical = NumericalModel()
            self.text = TextModel()
            self.classifier = nn.Linear(64, 1)  # خروجی: افزایش/کاهش قیمت
```
#### **توضیحات**:

-   مدل ترکیبی دو بخش را برای پردازش جداگانه داده‌ها دارد.
-   خروجی نهایی یک لایه سیگموئید برای پیش‌بینی احتمال افزایش/کاهش قیمت.

* * *

### **مرحله ۴: آموزش مدل**

#### ۱. تعریف تابع هزینه و بهینه‌ساز:

```python

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```
#### ۲. تقسیم داده‌ها:

-   **Train/Validation/Test**: 70%/15%/15%.
-   استفاده از `TimeSeriesSplit` برای جلوگیری از data leakage.

#### ۳. حلقه آموزش:

```python

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            numerical_data, text_data, labels = batch
            outputs = model(numerical_data, text_data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```
#### **توضیحات**:

-   استفاده از Early Stopping و Model Checkpoint برای جلوگیری از Overfitting.
-   ارزیابی با متریک‌های Accuracy, F1-Score, ROC-AUC.

* * *

### **مرحله ۵: شبیه‌سازی داده‌ها**

#### **داده‌های مصنوعی**:

```python

    import pandas as pd
    import numpy as np
    
    # داده‌های قیمت
    dates = pd.date_range("2023-01-01", periods=100)
    prices = np.cumprod(1 + np.random.normal(0, 0.01, 100))  # Random Walk
    df_prices = pd.DataFrame({"Date": dates, "Close": prices})
    
    # شاخص‌های تکنیکال
    df_prices["SMA_10"] = df_prices["Close"].rolling(10).mean()
    df_prices["RSI"] = np.random.randint(0, 100, 100)
    
    # عناوین خبری
    news_titles = [
        "شرکت X رکورد فروش جدیدی ثبت کرد",
        "کاهش قیمت نفت تأثیر منفی بر بازار گذاشت",
        "..."
    ]
    df_news = pd.DataFrame({"Date": dates[:50], "News_Title": news_titles * 10})  # فرض ۵۰ خبر
```
#### **ترکیب داده‌ها**:

```python

    df = pd.merge(df_prices, df_news, on="Date", how="left")
    df.fillna("", inplace=True)  # پر کردن خبرهای خالی
```
* * *

### **مرحله ۶: استفاده تجاری مدل**

#### **نکات کلیدی**:

1.  **سرعت پردازش**: استفاده از مدل‌های کوچکتر (مثل DistilBERT به جای BERT).
2.  **Deployment**: استفاده از `TorchScript` یا `ONNX` برای خروجی گرفتن از مدل و استفاده در محیط‌های تجاری.
3.  **Monitoring**: پایش دقت مدل در زمان واقعی و به‌روزرسانی دوره‌ای.
4.  **Risk Management**: ترکیب خروجی مدل با قوانین سرمایه‌گذاری (مثل Stop-Loss).

#### **مثال استفاده در API**:

```python

    from fastapi import FastAPI
    import torch
    
    app = FastAPI()
    model = torch.jit.load("hybrid_model.pt")
    
    @app.post("/predict")
    def predict(data: dict):
        # پردازش داده‌های ورودی
        prediction = model(numerical_data, text_data)
        return {"signal": "Buy" if prediction > 0.5 else "Sell"}
```
* * *

### **چالش‌ها و محدودیت‌ها**

1.  **داده‌های خبری**: تحلیل احساسات (Sentiment Analysis) خبرها دشوار است.
2.  **Random Walk بازار**: قیمت سهام ممکن است رفتار تصادفی داشته باشد.
3.  **Overfitting**: استفاده از داده‌های محدود می‌تواند منجر به Overfit شود.

* * *

### **نتیجه‌گیری**

این مدل ترکیبی قادر است الگوهای پیچیده‌ای را در داده‌های چند‌منبعی (قیمت، شاخص‌ها، خبرها) یاد بگیرد. با این حال، در محیط‌های تجاری باید با دقت استفاده شود و همراه با استراتژی‌های مدیریت ریسک باشد. برای بهبود عملکرد، می‌توان از داده‌های بیشتر، بهینه‌سازی هایپرپارامترها و استفاده از مدل‌های Transformer پیشرفته‌تر (مانند FinBERT) استفاده کرد.