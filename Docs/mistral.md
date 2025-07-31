متأسفانه، من نمی‌توانم کدهای پایتون را اجرا کنم یا به طور مستقیم مدل‌های یادگیری عمیق را پیاده‌سازی کنم. با این حال، می‌توانم یک راهنمای گام به گام برای ایجاد یک مدل یادگیری عمیق با استفاده از PyTorch برای پیش‌بینی حرکت روزانه قیمت سهام با استفاده از ترکیبی از قیمت، شاخص‌های فنی و عناوین خبری ارائه دهم. همچنین، می‌توانم نمونه‌ای از داده‌ها و کدهای مورد نیاز را به شما نشان دهم.

### گام 1: جمع‌آوری داده‌ها

اولین قدم جمع‌آوری داده‌های لازم است. داده‌ها می‌توانند شامل قیمت‌های تاریخی سهام، شاخص‌های فنی و عناوین خبری باشند.

#### نمونه داده‌ها

**داده‌های قیمت سهام و شاخص‌های فنی:**

| تاریخ | قیمت باز شدن | قیمت بسته شدن | بالاترین قیمت | پایین‌ترین قیمت | حجم | شاخص فنی 1 | شاخص فنی 2 |
|-------|---------------|----------------|---------------|----------------|------|--------------|--------------|
| 2023-01-01 | 100 | 105 | 110 | 99 | 1000000 | 0.5 | 0.7 |
| 2023-01-02 | 105 | 110 | 115 | 104 | 1200000 | 0.6 | 0.8 |
| 2023-01-03 | 110 | 108 | 112 | 107 | 900000 | 0.4 | 0.6 |

**داده‌های عناوین خبری:**

| تاریخ | عنوان خبر |
|-------|------------|
| 2023-01-01 | شرکت X اعلام کرد که سود سه ماهه خود را افزایش داده است |
| 2023-01-02 | شرکت Y با مشکل قانونی مواجه شده است |
| 2023-01-03 | شرکت Z اعلام کرد که محصول جدیدی را عرضه خواهد کرد |

### گام 2: پیش‌پردازش داده‌ها

داده‌ها باید پیش‌پردازش شوند تا بتوانند به عنوان ورودی به مدل یادگیری عمیق استفاده شوند. این شامل تبدیل داده‌های متنی به بردارهای عددی و نرمال‌سازی داده‌های عددی است.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# نمونه داده‌های قیمت سهام و شاخص‌های فنی
stock_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'open': [100, 105, 110],
    'close': [105, 110, 108],
    'high': [110, 115, 112],
    'low': [99, 104, 107],
    'volume': [1000000, 1200000, 900000],
    'technical_indicator_1': [0.5, 0.6, 0.4],
    'technical_indicator_2': [0.7, 0.8, 0.6]
})

# نمونه داده‌های عناوین خبری
news_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'title': [
        'شرکت X اعلام کرد که سود سه ماهه خود را افزایش داده است',
        'شرکت Y با مشکل قانونی مواجه شده است',
        'شرکت Z اعلام کرد که محصول جدیدی را عرضه خواهد کرد'
    ]
})

# نرمال‌سازی داده‌های عددی
scaler = MinMaxScaler()
stock_data[['open', 'close', 'high', 'low', 'volume', 'technical_indicator_1', 'technical_indicator_2']] = scaler.fit_transform(
    stock_data[['open', 'close', 'high', 'low', 'volume', 'technical_indicator_1', 'technical_indicator_2']]
)

# تبدیل داده‌های متنی به بردارهای عددی
vectorizer = TfidfVectorizer()
news_vectors = vectorizer.fit_transform(news_data['title'])
news_data = pd.DataFrame(news_vectors.toarray(), columns=vectorizer.get_feature_names_out())
```

### گام 3: ایجاد مدل یادگیری عمیق

در این گام، یک مدل یادگیری عمیق با استفاده از PyTorch ایجاد می‌کنیم. این مدل شامل لایه‌های مختلفی برای پردازش داده‌های عددی و متنی است.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# نمونه‌ای از داده‌های ورودی
input_size = stock_data.shape[1] + news_data.shape[1]
hidden_size = 64
output_size = 1  # پیش‌بینی حرکت قیمت سهام (بالا یا پایین)

model = StockPredictionModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### گام 4: آموزش مدل

در این گام، مدل را با استفاده از داده‌های آموزش می‌دهیم.

```python

# Code
# نمونه‌ای از داده‌های ورودی و خروجی
X = torch.tensor(pd.concat([stock_data, news_data], axis=1).values, dtype=torch.float32)
y = torch.tensor([[1], [1], [0]], dtype=torch.float32)  # نمونه‌ای از برچسب‌ها (1: بالا، 0: پایین)

# آموزش مدل
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
```

### گام 5: ارزیابی مدل

در این گام، مدل را با استفاده از داده‌های آزمون ارزیابی می‌کنیم.

```python

# Code
# نمونه‌ای از داده‌های آزمون
X_test = torch.tensor(pd.concat([stock_data, news_data], axis=1).values, dtype=torch.float32)
y_test = torch.tensor([[1], [1], [0]], dtype=torch.float32)

# ارزیابی مدل
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test)
    print(f'Test Loss: {loss.item():.4f}')
```

### گام 6: استفاده از مدل در محیط‌های تجاری

برای استفاده از مدل در محیط‌های تجاری، باید مدل را ذخیره کرده و در محیط‌های تولیدی بارگذاری کنیم.

```python

# Code
# ذخیره مدل
torch.save(model.state_dict(), 'stock_prediction_model.pth')

# بارگذاری مدل
model = StockPredictionModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('stock_prediction_model.pth'))
model.eval()
```

### نتیجه‌گیری

این راهنما یک دید کلی از چگونگی ایجاد یک مدل یادگیری عمیق برای پیش‌بینی حرکت روزانه قیمت سهام با استفاده از PyTorch ارائه می‌دهد. برای استفاده در محیط‌های تجاری، باید مدل را با داده‌های واقعی آموزش داده و ارزیابی کنید و همچنین باید زیرساخت‌های لازم برای استقرار مدل را فراهم کنید.