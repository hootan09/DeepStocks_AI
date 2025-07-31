با استفاده از pytorch و پایتون و deeplearning

 يك مدل يادگيري عميق كه پيشبيني روزانه حركت قيمت سهام را با استفاده از ﺗركيب

قيمت،شاخﺺهاي فني و عناوين خبري ميباشد،ارائه بده.

تمامی مراحل رو با جزپیات توضیح بده. همچنین مدل قابلیت استفاده در محیط های تجاری رو هم داشته باشه

* * *

## ساخت مدل یادگیری عمیق پیش‌بینی حرکت قیمت سهام با PyTorch

در این راهنما، به صورت جامع و گام به گام، نحوه ساخت یک مدل یادگیری عمیق برای پیش‌بینی حرکت روزانه قیمت سهام با استفاده از ترکیب **قیمت، شاخص‌های فنی و عناوین خبری** در **PyTorch** را توضیح می‌دهیم. این مدل با در نظر گرفتن قابلیت استفاده در **محیط‌های تجاری** طراحی خواهد شد.

### ۱. جمع‌آوری و آماده‌سازی داده‌ها (Data Collection and Preprocessing)

این گام حیاتی‌ترین بخش هر پروژه یادگیری عمیق است. برای این مدل، به سه نوع داده نیاز داریم:

-   **داده‌های تاریخی قیمت سهام:** شامل قیمت‌های باز شدن (Open)، بسته شدن (Close)، بالا (High)، پایین (Low) و حجم معاملات (Volume). این داده‌ها را می‌توانید از منابعی مانند Yahoo Finance، Alpha Vantage، یا پلتفرم‌های داده‌ای بورس خودتان (در ایران، شرکت مدیریت فناوری بورس تهران) دریافت کنید.
    
-   **شاخص‌های فنی (Technical Indicators):** این شاخص‌ها از داده‌های تاریخی قیمت محاسبه می‌شوند و می‌توانند الگوهای بازار را نشان دهند. شاخص‌های رایج شامل میانگین متحرک ساده (SMA)، میانگین متحرک نمایی (EMA)، قدرت نسبی (RSI)، MACD، باندهای بولینگر (Bollinger Bands) و... می‌باشند. می‌توانید از کتابخانه‌هایی مانند `talib` یا `stockstats` برای محاسبه این شاخص‌ها استفاده کنید.
    
-   **عناوین خبری (News Headlines):** برای تحلیل احساسات. این داده‌ها را می‌توان از APIsهای خبری مانند Google News API، News API، یا پلتفرم‌های خبری مالی تخصصی جمع‌آوری کرد. نکته مهم در اینجا، برچسب زمانی خبر است تا بتوان آن را با داده‌های قیمت همگام‌سازی کرد.
    

**مراحل آماده‌سازی:**

1.  **همگام‌سازی زمانی:** اطمینان حاصل کنید که تمام داده‌ها بر اساس زمان همگام‌سازی شده‌اند. برای مثال، هر سطر داده باید مربوط به یک روز خاص باشد.
    
2.  **پاک‌سازی داده‌ها:**
    
    -   **داده‌های گم‌شده (Missing Values):** در صورت وجود، با روش‌هایی مانند میانگین‌گیری، میانه، یا پیش‌بینی (Interpolation) جایگزین شوند.
        
    -   **داده‌های پرت (Outliers):** بررسی و در صورت نیاز، حذف یا تعدیل شوند.
        
3.  **مهندسی ویژگی (Feature Engineering):**
    
    -   **ویژگی‌های قیمت:** قیمت‌های باز شدن، بسته شدن، بالا، پایین و حجم معاملات می‌توانند به صورت مستقیم استفاده شوند. می‌توانید ویژگی‌هایی مانند بازده روزانه یا تغییرات درصد قیمت را نیز اضافه کنید.
        
    -   **ویژگی‌های شاخص فنی:** شاخص‌های محاسبه شده به عنوان ورودی‌های عددی به مدل داده می‌شوند.
        
    -   **ویژگی‌های متنی (News Headlines):**
        
        -   **توکن‌سازی (Tokenization):** تبدیل جملات به کلمات یا توکن‌ها.
            
        -   **حذف کلمات توقف (Stop Word Removal):** حذف کلمات رایج و بی‌اهمیت (مانند "یک"، "و"، "است").
            
        -   **ریشه‌یابی (Stemming) یا لماتیزاسیون (Lemmatization):** کاهش کلمات به ریشه اصلی آنها.
            
        -   **نمایش برداری (Vectorization):** تبدیل متن به بردارهای عددی قابل فهم برای مدل. روش‌های رایج شامل:
            
            -   **TF-IDF (Term Frequency-Inverse Document Frequency):** وزن‌دهی به کلمات بر اساس فراوانی آنها در یک سند و نادر بودن آنها در کل مجموعه.
                
            -   **Word Embeddings (مانند Word2Vec، GloVe، FastText):** نمایش کلمات در یک فضای برداری به گونه‌ای که کلمات با معنای مشابه به هم نزدیک باشند. این روش‌ها به مدل اجازه می‌دهند روابط معنایی بین کلمات را درک کند. برای زبان فارسی، می‌توانید از مدل‌های آماده Word Embeddings یا خودتان مدل را آموزش دهید.
                
            -   **BERT، XLM-RoBERTa (برای زبان فارسی):** مدل‌های پیش‌آموزش‌دیده ترانسفورمر که نمایش‌های معنایی قوی‌تری از متن ارائه می‌دهند. استفاده از این مدل‌ها به دلیل پیچیدگی بالا، نیاز به منابع محاسباتی بیشتری دارد.
                
        -   **تحلیل احساسات (Sentiment Analysis):** اختصاص نمره احساسی (مثبت، منفی، خنثی) به هر عنوان خبری. می‌توانید از مدل‌های پیش‌آموزش‌دیده تحلیل احساسات (مانند VADER برای انگلیسی) یا یک مدل سفارشی آموزش‌داده شده بر روی داده‌های فارسی استفاده کنید. نمره احساسات به عنوان یک ویژگی عددی به مدل اضافه می‌شود.
            
4.  **نرمال‌سازی (Normalization) / استانداردسازی (Standardization):** مقیاس‌بندی ویژگی‌های عددی به یک محدوده خاص (مثلاً 0 تا 1) یا با میانگین 0 و انحراف معیار 1. این کار به مدل کمک می‌کند تا سریع‌تر همگرا شود و عملکرد بهتری داشته باشد.
    
5.  **تقسیم داده‌ها:** داده‌ها را به مجموعه آموزش (Training Set)، اعتبارسنجی (Validation Set) و آزمایش (Test Set) تقسیم کنید. برای داده‌های سری زمانی، تقسیم‌بندی باید به صورت **زمانی** باشد (یعنی داده‌های قدیمی‌تر برای آموزش و داده‌های جدیدتر برای آزمایش).
    

### ۲. طراحی معماری مدل (Model Architecture Design)

با توجه به ماهیت داده‌ها، یک معماری چندورودی (Multi-Input) مناسب است که بتواند هر سه نوع داده را به طور جداگانه پردازش کرده و سپس ترکیب کند.

-   **ورودی داده‌های عددی (قیمت و شاخص‌های فنی):**
    
    -   **شبکه عصبی بازگشتی (Recurrent Neural Network - RNN):** مانند **LSTM (Long Short-Term Memory)** یا **GRU (Gated Recurrent Unit)** برای ضبط وابستگی‌های زمانی در سری‌های قیمت و شاخص‌های فنی. LSTMها برای مقابله با مشکل گرادیان ناپدید شونده/منفجر شونده در RNNهای سنتی مناسب‌ترند.
        
    -   **شبکه کانولوشنی یک‌بعدی (1D Convolutional Neural Network - 1D CNN):** برای استخراج الگوهای محلی از سری‌های زمانی.
        
-   **ورودی داده‌های متنی (عناوین خبری):**
    
    -   **مدل‌های Transformer (مانند BERT یا RoBERTa):** اگر از نمایش‌های برداری پیشرفته مانند BERT استفاده می‌کنید، می‌توانید لایه‌های خروجی این مدل‌ها را به عنوان ویژگی‌های متن به مدل اصلی خود اضافه کنید.
        
    -   **LSTM / GRU:** پس از تبدیل عناوین خبری به بردارهای Word Embedding، می‌توان از LSTM/GRU برای پردازش توالی کلمات و استخراج ویژگی‌های معنایی استفاده کرد.
        
    -   **CNN (با فیلترهای مختلف):** برای استخراج ویژگی‌های محلی از بردارهای کلمات.
        
-   **ترکیب ویژگی‌ها و لایه‌های خروجی:**
    
    -   **ادغام (Concatenation):** خروجی نهایی هر شاخه (قیمت/شاخص‌های فنی و خبر) را به هم پیوند دهید.
        
    -   **لایه‌های متراکم (Dense/Fully Connected Layers):** پس از ادغام، چند لایه متراکم با توابع فعال‌سازی (مانند ReLU) برای یادگیری روابط پیچیده بین ویژگی‌های ترکیب شده.
        
    -   **لایه خروجی (Output Layer):**
        
        -   **برای طبقه‌بندی (پیش‌بینی حرکت: بالا/پایین/خنثی):** یک لایه متراکم با `softmax` (برای چند دسته) یا `sigmoid` (برای دو دسته) و تعداد نورون‌ها برابر با تعداد دسته‌ها.
            
        -   **برای رگرسیون (پیش‌بینی تغییر درصد قیمت):** یک لایه متراکم با یک نورون و بدون تابع فعال‌سازی (یا تابع فعال‌سازی خطی).
            

**مثال معماری (ترکیبی از LSTM و CNN برای متن):**

```Python

import torch
    import torch.nn as nn
    
    class StockMovementPredictor(nn.Module):
        def __init__(self, numerical_features_dim, vocab_size, embedding_dim, hidden_size_lstm, num_classes):
            super(StockMovementPredictor, self).__init__()
    
            # --- Numerical Data Branch (Price & Technical Indicators) ---
            self.lstm = nn.LSTM(input_size=numerical_features_dim, hidden_size=hidden_size_lstm, batch_first=True)
            self.fc_numerical = nn.Linear(hidden_size_lstm, 64) # Example intermediate layer
    
            # --- Text Data Branch (News Headlines) ---
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            # 1D CNN for text
            self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.global_max_pool = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
            self.fc_text = nn.Linear(128, 64) # Example intermediate layer
    
            # --- Combined Layers ---
            self.dropout = nn.Dropout(0.5)
            self.fc_combined = nn.Linear(64 + 64, 32) # Sum of output dimensions from each branch
            self.output_layer = nn.Linear(32, num_classes) # For classification (e.g., 3 classes: Up, Down, Neutral)
    
        def forward(self, numerical_input, text_input):
            # Numerical branch
            lstm_out, _ = self.lstm(numerical_input)
            # Use the last hidden state of the LSTM or flatten and apply pooling
            numerical_features = self.fc_numerical(lstm_out[:, -1, :]) # Taking the last time step output
    
            # Text branch
            embedded_text = self.embedding(text_input) # text_input should be tokenized indices
            embedded_text = embedded_text.permute(0, 2, 1) # Change to (batch_size, embedding_dim, sequence_length)
            conv_out = self.conv1d(embedded_text)
            conv_out = self.relu(conv_out)
            text_features = self.global_max_pool(conv_out).squeeze(2) # Remove the last dimension
            text_features = self.fc_text(text_features)
    
            # Combine features
            combined_features = torch.cat((numerical_features, text_features), dim=1)
            combined_features = self.dropout(combined_features)
            
            # Final layers
            x = self.fc_combined(combined_features)
            x = self.relu(x)
            x = self.dropout(x)
            output = self.output_layer(x)
    
            return output
```
### ۳. آموزش مدل (Model Training)

**گام‌های کلیدی:**

1.  **تعریف تابع زیان (Loss Function):**
    
    -   **برای طبقه‌بندی:** `nn.CrossEntropyLoss` (اگر خروجی مدل logits باشند) یا `nn.BCELoss` (اگر خروجی مدل احتمالات بین 0 و 1 باشند و دو دسته داشته باشیم).
        
    -   **برای رگرسیون:** `nn.MSELoss` (Mean Squared Error) یا `nn.L1Loss` (Mean Absolute Error).
        
    -  **تعریف بهینه‌ساز (Optimizer):** `torch.optim.Adam`, `torch.optim.SGD`, `torch.optim.RMSprop`. `Adam` معمولاً یک انتخاب خوب برای شروع است.
    
    -  **انتخاب دستگاه (Device):** آموزش مدل بر روی GPU (اگر در دسترس باشد) سرعت را به شدت افزایش می‌دهد.
    
```Python
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
```
2.  **حلقه آموزش (Training Loop):**
    
    -   **DataLoaders:** از `torch.utils.data.DataLoader` برای ایجاد بسته‌های (Batches) داده استفاده کنید. این کار به مدیریت حافظه و سرعت آموزش کمک می‌کند.
        
    -   **Epochs:** تعداد دفعاتی که کل مجموعه داده آموزشی از مدل عبور می‌کند.
        
    -   **Batch Size:** تعداد نمونه‌هایی که در هر مرحله (Iteration) به مدل داده می‌شوند.
        
    -   **Forward Pass:** داده‌های ورودی را به مدل می‌دهید و خروجی را دریافت می‌کنید.
        
    -   **Calculate Loss:** تابع زیان را بر روی خروجی مدل و برچسب‌های واقعی محاسبه می‌کنید.
        
    -   **Backward Pass:** گرادیان‌ها (Gradients) را محاسبه می‌کنید (`loss.backward()`).
        
    -   **Optimizer Step:** وزن‌های مدل را با استفاده از گرادیان‌ها و بهینه‌ساز به‌روزرسانی می‌کنید (`optimizer.step()`).
        
    -   **Zero Grads:** گرادیان‌ها را صفر می‌کنید (`optimizer.zero_grad()`) تا از انباشتگی آنها در مرحله بعدی جلوگیری شود.
        
3.  **اعتبارسنجی (Validation):** در طول آموزش، به صورت دوره‌ای مدل را روی مجموعه اعتبارسنجی ارزیابی کنید تا از overfitting جلوگیری شود و بهترین مدل را ذخیره کنید.
    
4.  **زمان‌بندی نرخ یادگیری (Learning Rate Scheduler):** برای تنظیم پویا نرخ یادگیری در طول آموزش که می‌تواند به همگرایی بهتر کمک کند.
    

```Python

    # Assuming you have your DataLoader instances: train_loader, val_loader
    # And your model, loss_function, optimizer defined
    
    num_epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        train_loss = 0
        for batch_idx, (numerical_data, text_data, labels) in enumerate(train_loader):
            numerical_data, text_data, labels = numerical_data.to(device), text_data.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(numerical_data, text_data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")
    
        # --- Validation ---
        model.eval() # Set model to evaluation mode
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
    
        with torch.no_grad(): # Disable gradient calculations for validation
            for numerical_data, text_data, labels in val_loader:
                numerical_data, text_data, labels = numerical_data.to(device), text_data.to(device), labels.to(device)
                outputs = model(numerical_data, text_data)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
    
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
    
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct_predictions / total_predictions
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_stock_predictor.pth')
            print("Saved best model!")
```
### ۴. ارزیابی مدل (Model Evaluation)

پس از آموزش، مدل را بر روی مجموعه **آزمایش (Test Set)** که هرگز در طول آموزش دیده نشده است، ارزیابی کنید تا عملکرد واقعی آن را بسنجید.

-   **معیارهای ارزیابی برای طبقه‌بندی:**
    
    -   **دقت (Accuracy):** نسبت پیش‌بینی‌های صحیح به کل پیش‌بینی‌ها.
        
    -   **توالی (Precision):** از میان پیش‌بینی‌های مثبت، چه تعداد واقعاً مثبت بوده‌اند.
        
    -   **بازیابی (Recall):** از میان موارد مثبت واقعی، چه تعداد به درستی پیش‌بینی شده‌اند.
        
    -   **F1-Score:** میانگین هارمونیک Precision و Recall.
        
    -   **ماتریس درهم‌ریختگی (Confusion Matrix):** یک جدول برای تجسم عملکرد مدل طبقه‌بندی.
        
-   **معیارهای ارزیابی برای رگرسیون:**
    
    -   **MAE (Mean Absolute Error)**
        
    -   **RMSE (Root Mean Squared Error)**
        
    -   **R-squared**
        

```Python

# Load the best model
    model.load_state_dict(torch.load('best_stock_predictor.pth'))
    model.eval() # Set to evaluation mode
    
    test_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for numerical_data, text_data, labels in test_loader:
            numerical_data, text_data, labels = numerical_data.to(device), text_data.to(device), labels.to(device)
            outputs = model(numerical_data, text_data)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
    
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
```
### ۵. استقرار و استفاده در محیط تجاری (Deployment and Commercial Use)

برای استفاده از مدل در محیط‌های تجاری، به موارد زیر نیاز دارید:

1.  **پایپ‌لاین داده زنده (Live Data Pipeline):**
    
    -   **جمع‌آوری داده‌های لحظه‌ای:** سیستمی برای جمع‌آوری قیمت‌های لحظه‌ای سهام، شاخص‌های فنی و عناوین خبری در زمان واقعی یا با تأخیر کم. این ممکن است شامل اتصال به APIsهای WebSocket یا پروتکل‌های دیگر باشد.
        
    -   **پردازش داده‌های لحظه‌ای:** داده‌های جدید باید با همان روشی که داده‌های آموزشی آماده شدند، نرمال‌سازی و پیش‌پردازش شوند (مثلاً، شاخص‌های فنی جدید محاسبه شوند، عناوین خبری جدید توکن‌سازی و تحلیل احساسات شوند).
        
    -   **مدل‌سازی ویژگی‌های متنی:** اگر از مدل‌های پیش‌آموزش‌دیده ترانسفورمر (مانند BERT) استفاده می‌کنید، باید این مدل‌ها را در محیط استقرار خود نیز لود کنید.
        
2.  **API برای پیش‌بینی (Prediction API):**
    
    -   یک سرویس RESTful API (مثلاً با استفاده از Flask، FastAPI یا Django) بسازید که درخواست‌های ورودی (داده‌های قیمت، شاخص، خبر برای یک سهام خاص) را دریافت کند.
        
    -   مدل آموزش‌دیده را لود کنید.
        
    -   داده‌های ورودی را به فرمت Tensor تبدیل کرده و به دستگاه مناسب (CPU/GPU) بفرستید.
        
    -   پیش‌بینی را از مدل دریافت کنید.
        
    -   پیش‌بینی را در قالب JSON یا فرمت مناسب دیگری به کاربر بازگردانید.
        
3.  **مانیتورینگ و لاگینگ (Monitoring and Logging):**
    
    -   **عملکرد مدل:** عملکرد مدل در محیط واقعی را به طور مداوم نظارت کنید. آیا دقت آن در طول زمان کاهش می‌یابد؟ (Drift Data / Model Decay).
        
    -   **کارایی سیستم:** مانیتورینگ مصرف منابع (CPU، RAM، GPU)، زمان پاسخگویی API و خطاها.
        
    -   **لاگ‌برداری:** ثبت درخواست‌ها، پیش‌بینی‌ها و هرگونه خطایی برای اشکال‌زدایی و تحلیل‌های بعدی.
        
4.  **مدیریت مدل (Model Management):**
    
    -   **نسخه‌بندی مدل (Model Versioning):** هر نسخه از مدل را با دقت ردیابی کنید.
        
    -   **بازآموزی (Retraining):** قیمت سهام و اخبار به طور مداوم در حال تغییر هستند. مدل باید به صورت دوره‌ای با داده‌های جدیدتر بازآموزی شود تا مرتبط و دقیق باقی بماند. این کار می‌تواند به صورت خودکار یا دستی انجام شود.
        
    -   **آزمایش A/B (A/B Testing):** اگر نسخه‌های جدیدی از مدل را استقرار می‌دهید، عملکرد آن‌ها را با نسخه فعلی مقایسه کنید.
        
5.  **امنیت (Security):**
    
    -   **احراز هویت و مجوز (Authentication and Authorization):** اطمینان از اینکه فقط کاربران مجاز می‌توانند به API شما دسترسی داشته باشند.
        
    -   **محافظت در برابر حملات (Protection against attacks):** اقدامات امنیتی استاندارد وب را برای API خود پیاده‌سازی کنید.
        
6.  **اسکال‌پذیری (Scalability):**
    
    -   سیستم را به گونه‌ای طراحی کنید که بتواند با افزایش درخواست‌ها مقابله کند. این ممکن است شامل استفاده از Load Balancerها، Microservices، یا کانتینری‌سازی (مانند Docker و Kubernetes) باشد.
        

**مثال اسکلت یک API با FastAPI:**

```Python

# pip install fastapi uvicorn torch pandas numpy transformers # if using BERT
    
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification # if using pre-trained sentiment model
    
    # Assuming your StockMovementPredictor class is defined in model.py
    from model import StockMovementPredictor # You would put your model class here
    
    app = FastAPI()
    
    # Configuration and Model Loading
    MODEL_PATH = "best_stock_predictor.pth"
    NUM_CLASSES = 3 # Up, Down, Neutral (example)
    NUMERICAL_FEATURES_DIM = 10 # Example: (Close, Open, High, Low, Volume, RSI, MACD_hist, etc.)
    VOCAB_SIZE = 10000 # Example, based on your preprocessed text data
    EMBEDDING_DIM = 100 # Example
    HIDDEN_SIZE_LSTM = 128 # Example
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your model
    try:
        model = StockMovementPredictor(NUMERICAL_FEATURES_DIM, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE_LSTM, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Consider raising an error or exiting if model loading fails
        model = None
    
    # Load your tokenizer and any other necessary preprocessing tools (e.g., normalizers)
    # tokenizer = AutoTokenizer.from_pretrained("your_persian_tokenizer") # If using pre-trained
    # sentiment_model = AutoModelForSequenceClassification.from_pretrained("your_persian_sentiment_model") # If using pre-trained
    
    
    # Define input data structure for the API
    class PredictionRequest(BaseModel):
        numerical_data: list[list[float]] # Nested list for time series (e.g., [[val1, val2, ...], [val1, val2, ...]])
        news_headline: str
    
    @app.post("/predict_stock_movement/")
    async def predict_stock_movement(request: PredictionRequest):
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Server error.")
    
        try:
            # 1. Preprocess Numerical Data
            numerical_input = torch.tensor(request.numerical_data, dtype=torch.float32).unsqueeze(0).to(device)
            # Ensure numerical_input has shape (batch_size, sequence_length, numerical_features_dim)
            # For a single prediction, batch_size=1
    
            # 2. Preprocess Text Data
            # This part needs your actual text preprocessing logic (tokenization, vectorization, sentiment)
            # For demonstration, let's assume a simple tokenization and then mapping to integer IDs
            # In a real scenario, you'd apply the same tokenization/embedding process as in training.
            
            # Example: Simple tokenization and integer mapping (replace with your actual logic)
            # This is a placeholder and assumes you have a way to convert words to vocabulary IDs
            # For a production system, you'd use your trained Word2Vec/GloVe or BERT pipeline
            tokens = request.news_headline.lower().split()
            # Assume a dummy vocabulary for this example
            dummy_vocab = {word: i for i, word in enumerate(set(tokens))} # NOT for production!
            
            token_ids = [dummy_vocab.get(token, 0) for token in tokens] # Map words to IDs
            # Padding or truncation might be needed based on your model's input
            # For simplicity, let's just take the first few tokens
            max_seq_len = 20 # Example max sequence length
            if len(token_ids) < max_seq_len:
                token_ids.extend([0] * (max_seq_len - len(token_ids)))
            else:
                token_ids = token_ids[:max_seq_len]
    
            text_input = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    
            # 3. Make Prediction
            with torch.no_grad():
                output = model(numerical_input, text_input)
                probabilities = torch.softmax(output, dim=1)
                _, predicted_class_idx = torch.max(probabilities, 1)
    
            # Map class index back to human-readable label
            # Example: 0 -> 'Down', 1 -> 'Neutral', 2 -> 'Up'
            movement_labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}
            predicted_movement = movement_labels.get(predicted_class_idx.item(), 'Unknown')
    
            return {"predicted_movement": predicted_movement, "probabilities": probabilities.tolist()}
    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    # To run this API:
    # 1. Save the model definition (StockMovementPredictor) in a file like model.py
    # 2. Save this FastAPI code in a file like main.py
    # 3. Run from your terminal: uvicorn main:app --reload
    # 4. Access the API at http://127.0.0.1:8000/docs for interactive documentation.
```
این راهنما یک چارچوب جامع برای ساخت و استقرار مدل شما ارائه می‌دهد. پیاده‌سازی جزئیات هر بخش، به ویژه جمع‌آوری و پیش‌پردازش داده‌ها و تنظیم دقیق معماری مدل، نیازمند تلاش و تخصص است. موفق باشید!

* * *

آیا در مورد هر یک از این مراحل نیاز به توضیح یا جزئیات بیشتری دارید؟ یا در مورد بخش خاصی از کد سوالی دارید؟

با مثال چند نمونه از داده های مورد نیاز رو دسته بندی کن و به مدل بده و خروجی رو شبیه سازی کن.

داده های نمونه و sample رو به منم نشون بده تا ببینم منظورت چیه و چه شکلی هست.

* * *

حله! برای اینکه تصویر واضح‌تری از داده‌های مورد نیاز و نحوه ورودی دادن به مدل پیدا کنیم، بیایید چند نمونه داده فرضی ایجاد و نحوه پردازش و فرمت‌بندی اونها رو برای ورودی مدل شبیه‌سازی کنیم. در نهایت هم خروجی مدل رو با استفاده از این داده‌های نمونه شبیه‌سازی می‌کنیم.

### ۱. دسته‌بندی و تولید داده‌های نمونه

همونطور که پیش‌تر اشاره شد، ما سه نوع داده اصلی داریم: **داده‌های عددی (قیمت و شاخص‌های فنی)**، و **داده‌های متنی (عناوین خبری)**. برای سادگی، فرض می‌کنیم پیش‌پردازش‌های اولیه انجام شده و داده‌ها آماده ورود به مدل هستند.

#### ۱.۱. داده‌های قیمت و شاخص‌های فنی (Numerical Data)

این داده‌ها معمولاً یک **سری زمانی** هستند. برای هر روز، ما چندین ویژگی عددی داریم. فرض کنید می‌خواهیم حرکت قیمت رو بر اساس ۵ روز گذشته پیش‌بینی کنیم. پس هر نمونه ورودی شامل داده‌های ۵ روز متوالی خواهد بود.

**ویژگی‌های نمونه برای هر روز:**

-   **قیمت بسته شدن (Close Price):** `100.5`
    
-   **حجم معاملات (Volume):** `1,200,000`
    
-   **شاخص RSI:** `65.2`
    
-   **شاخص MACD Line:** `2.1`
    
-   **شاخص MACD Signal:** `1.8`
    

**نمونه داده‌های ۵ روزه (یک نمونه ورودی به مدل):**


 
| Day       | Close Price | Volume   | RSI   | MACD Line | MACD Signal |
|-----------|-------------|----------|-------|-----------|-------------|
| **Day -4** | 98.2        | 1.1M     | 58.0  | 1.5       | 1.3         |
| **Day -3** | 99.0        | 1.3M     | 61.5  | 1.7       | 1.5         |
| **Day -2** | 100.1       | 1.0M     | 63.0  | 1.9       | 1.6         |
| **Day -1** | 100.5       | 1.2M     | 65.2  | 2.1       | 1.8         |
| **Day 0**  | 101.3       | 1.5M     | 68.0  | 2.5       | 2.0         |

**توضیح:** در اینجا "Day 0" آخرین روزی است که داده‌های آن را داریم و می‌خواهیم حرکت "Day +1" را پیش‌بینی کنیم.

**فرم ورودی به PyTorch:** این داده‌ها باید به یک تنسور (Tensor) تبدیل شوند. اگر یک نمونه را در نظر بگیریم، شکل آن `(sequence_length, num_numerical_features)` خواهد بود. برای `batch_size=1`، شکل کلی ورودی `(1, sequence_length, num_numerical_features)` است.

```Python

import torch
    import numpy as np
    
    # فرض کنید اینها داده‌های ۵ روز متوالی برای یک سهم هستند (نرمال شده)
    # هر ردیف مربوط به یک روز و هر ستون یک ویژگی است.
    sample_numerical_data = np.array([
        [0.5, 0.4, 0.6, 0.3, 0.5],  # Day -4 (example normalized values)
        [0.6, 0.5, 0.7, 0.4, 0.6],  # Day -3
        [0.7, 0.3, 0.8, 0.5, 0.7],  # Day -2
        [0.8, 0.4, 0.9, 0.6, 0.8],  # Day -1
        [0.9, 0.6, 1.0, 0.7, 0.9]   # Day 0
    ])
    
    # تبدیل به تنسور PyTorch
    numerical_input_tensor = torch.tensor(sample_numerical_data, dtype=torch.float32)
    # اضافه کردن بُعد batch (batch_size=1)
    numerical_input_tensor = numerical_input_tensor.unsqueeze(0) 
    
    print("Sample Numerical Data (Numpy):\n", sample_numerical_data)
    print("\nNumerical Input Tensor Shape:", numerical_input_tensor.shape)
    print("Numerical Input Tensor:\n", numerical_input_tensor)
```
**خروجی کد (Sample Numerical Data):**
```
Sample Numerical Data (Numpy):
     [[0.5 0.4 0.6 0.3 0.5]
     [0.6 0.5 0.7 0.4 0.6]
     [0.7 0.3 0.8 0.5 0.7]
     [0.8 0.4 0.9 0.6 0.8]
     [0.9 0.6 1.0 0.7 0.9]]
    
    Numerical Input Tensor Shape: torch.Size([1, 5, 5])
    Numerical Input Tensor:
     tensor([[[0.5000, 0.4000, 0.6000, 0.3000, 0.5000],
              [0.6000, 0.5000, 0.7000, 0.4000, 0.6000],
              [0.7000, 0.3000, 0.8000, 0.5000, 0.7000],
              [0.8000, 0.4000, 0.9000, 0.6000, 0.8000],
              [0.9000, 0.6000, 1.0000, 0.7000, 0.9000]]])
```
در اینجا `torch.Size([1, 5, 5])` به معنای `(batch_size, sequence_length, num_features)` است.

#### ۱.۲. داده‌های متنی (News Headlines)

این داده‌ها شامل عناوین خبری مربوط به سهام یا بازار در روز مورد نظر (Day 0) هستند. برای مدل، هر عنوان خبر باید به یک توالی از IDهای عددی تبدیل شود (پس از توکن‌سازی و نگاشت به واژگان). همچنین، نمره احساسات نیز می‌تواند به عنوان یک ویژگی عددی مجزا اضافه شود یا توسط مدل پردازش شود.

**نمونه عنوان خبری:** "تصویب طرح جدید دولت، بورس را به وجد آورد."

**مراحل پیش‌پردازش برای ورودی مدل:**

1.  **توکن‌سازی:** تبدیل جمله به کلمات ("تصویب", "طرح", "جدید", "دولت", "بورس", "به", "وجد", "آورد").
    
2.  **نگاشت به ID واژگان:** هر کلمه به یک ID عددی منحصر به فرد (بر اساس واژگان آموزش‌دیده) نگاشت می‌شود.
    
3.  **پدینگ/برش (Padding/Truncation):** تمامی توالی‌های توکن باید طول یکسانی داشته باشند. توالی‌های کوتاه‌تر با صفر پدینگ می‌شوند و توالی‌های بلندتر برش داده می‌شوند.
    
4.  **تحلیل احساسات (Sentiment Analysis):** مثلاً نمره احساسات `0.8` (مثبت).
    

**نمونه ورودی متنی (پس از پیش‌پردازش و تبدیل به ID):** فرض کنید طول ثابت توالی (Max Sequence Length) برای متن ۱۰ باشد و IDهای فرضی به صورت زیر باشند:

-   تصویب: 12
    
-   طرح: 23
    
-   جدید: 34
    
-   دولت: 45
    
-   بورس: 56
    
-   به: 67
    
-   وجد: 78
    
-   آورد: 89
    
-   (توکن‌های پدینگ): 0
    

**توالی IDها:** `[12, 23, 34, 45, 56, 67, 78, 89, 0, 0]` (یا اگر بیش از یک خبر در روز باشد، میانگین یا ترکیب احساسات آن‌ها) **نمره احساسات:** `0.8` (این می‌تواند به عنوان یک ویژگی عددی به بخش عددی اضافه شود یا در بخش متنی به نوعی ترکیب شود)

**فرم ورودی به PyTorch:** این داده‌ها نیز باید به یک تنسور تبدیل شوند. برای `batch_size=1`، شکل ورودی `(1, max_sequence_length)` خواهد بود.

```Python

# فرض کنید اینها توکن‌های یک عنوان خبری هستند که به ID تبدیل شده‌اند (با پدینگ)
    sample_text_ids = np.array([12, 23, 34, 45, 56, 67, 78, 89, 0, 0])
    
    # تبدیل به تنسور PyTorch (نوع long برای embeddingها)
    text_input_tensor = torch.tensor(sample_text_ids, dtype=torch.long)
    # اضافه کردن بُعد batch (batch_size=1)
    text_input_tensor = text_input_tensor.unsqueeze(0)
    
    print("\nSample Text IDs (Numpy):\n", sample_text_ids)
    print("\nText Input Tensor Shape:", text_input_tensor.shape)
    print("Text Input Tensor:\n", text_input_tensor)
```
**خروجی کد (Sample Text Data):**
```
Sample Text IDs (Numpy):
     [12 23 34 45 56 67 78 89  0  0]
    
    Text Input Tensor Shape: torch.Size([1, 10])
    Text Input Tensor:
     tensor([[12, 23, 34, 45, 56, 67, 78, 89,  0,  0]])
```
در اینجا `torch.Size([1, 10])` به معنای `(batch_size, sequence_length_of_text)` است.

### ۲. شبیه‌سازی مدل و خروجی

حالا با استفاده از کلاس مدل `StockMovementPredictor` که قبلاً تعریف کردیم، ورودی‌های نمونه رو بهش میدیم و خروجی رو شبیه‌سازی می‌کنیم.

**یادآوری معماری مدل:**

```Python

import torch
    import torch.nn as nn
    
    class StockMovementPredictor(nn.Module):
        def __init__(self, numerical_features_dim, vocab_size, embedding_dim, hidden_size_lstm, num_classes):
            super(StockMovementPredictor, self).__init__()
    
            # Numerical Data Branch (Price & Technical Indicators)
            self.lstm = nn.LSTM(input_size=numerical_features_dim, hidden_size=hidden_size_lstm, batch_first=True)
            self.fc_numerical = nn.Linear(hidden_size_lstm, 64) 
    
            # Text Data Branch (News Headlines)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.global_max_pool = nn.AdaptiveMaxPool1d(1) 
            self.fc_text = nn.Linear(128, 64) 
    
            # Combined Layers
            self.dropout = nn.Dropout(0.5)
            self.fc_combined = nn.Linear(64 + 64, 32) 
            self.output_layer = nn.Linear(32, num_classes) 
    
        def forward(self, numerical_input, text_input):
            # Numerical branch
            lstm_out, _ = self.lstm(numerical_input)
            numerical_features = self.fc_numerical(lstm_out[:, -1, :]) 
    
            # Text branch
            embedded_text = self.embedding(text_input) 
            embedded_text = embedded_text.permute(0, 2, 1) 
            conv_out = self.conv1d(embedded_text)
            conv_out = self.relu(conv_out)
            text_features = self.global_max_pool(conv_out).squeeze(2) 
            text_features = self.fc_text(text_features)
    
            # Combine features
            combined_features = torch.cat((numerical_features, text_features), dim=1)
            combined_features = self.dropout(combined_features)
            
            # Final layers
            x = self.fc_combined(combined_features)
            x = self.relu(x)
            x = self.dropout(x)
            output = self.output_layer(x)
    
            return output
```
**شبیه‌سازی فرآیند پیش‌بینی:**

```Python
# Code
# تعریف پارامترهای فرضی برای مدل
    numerical_features_dim = 5  # 5 ویژگی عددی (Close, Volume, RSI, MACD_Line, MACD_Signal)
    vocab_size = 10000          # اندازه واژگان فرضی
    embedding_dim = 100         # ابعاد جاسازی کلمه
    hidden_size_lstm = 128      # اندازه لایه مخفی LSTM
    num_classes = 3             # 3 کلاس خروجی: 0 (کاهش), 1 (بدون تغییر), 2 (افزایش)
    
    # ساخت نمونه‌ای از مدل
    model = StockMovementPredictor(numerical_features_dim, vocab_size, embedding_dim, hidden_size_lstm, num_classes)
    model.eval() # مدل را در حالت ارزیابی قرار می‌دهیم (برای غیرفعال کردن Dropout و Batch Norm)
    
    # (اختیاری) اگر مدل را قبلاً آموزش داده و ذخیره کرده‌اید:
    # model.load_state_dict(torch.load('best_stock_predictor.pth'))
    
    # انتقال مدل به CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # انتقال داده‌های ورودی به CPU/GPU
    numerical_input_tensor = numerical_input_tensor.to(device)
    text_input_tensor = text_input_tensor.to(device)
    
    # انجام پیش‌بینی
    with torch.no_grad(): # در زمان پیش‌بینی، نیازی به محاسبه گرادیان نداریم
        output = model(numerical_input_tensor, text_input_tensor)
    
    # تفسیر خروجی
    # برای طبقه‌بندی، معمولاً از softmax برای تبدیل logits به احتمالات استفاده می‌کنیم
    probabilities = torch.softmax(output, dim=1)
    
    # پیدا کردن کلاسی با بالاترین احتمال
    _, predicted_class_idx = torch.max(probabilities, 1)
    
    # نگاشت شاخص کلاس به برچسب‌های قابل فهم
    movement_labels = {0: 'کاهش', 1: 'بدون تغییر', 2: 'افزایش'}
    predicted_movement = movement_labels.get(predicted_class_idx.item(), 'نامشخص')
    
    print("\n--- Model Output Simulation ---")
    print("Raw Model Output (Logits):\n", output)
    print("\nPredicted Probabilities (Softmax):\n", probabilities)
    print("\nPredicted Class Index:", predicted_class_idx.item())
    print(f"Predicted Stock Movement for next day: '{predicted_movement}'")
```
**خروجی شبیه‌سازی مدل:** (مقادیر خروجی مدل به دلیل تصادفی بودن وزن‌های اولیه مدل، هر بار متفاوت خواهد بود، اما ساختار آن یکسان است.)
```
--- Model Output Simulation ---
    Raw Model Output (Logits):
     tensor([[-0.1234,  0.5678, -0.3456]], device='cuda:0') # Example random logits
    
    Predicted Probabilities (Softmax):
     tensor([[0.2456, 0.5009, 0.2535]], device='cuda:0') # Example probabilities summing to 1
    
    Predicted Class Index: 1
    Predicted Stock Movement for next day: 'بدون تغییر'
```
**توضیح خروجی:**

-   **`Raw Model Output (Logits)`:** این‌ها مقادیر خام (قبل از اعمال تابع فعال‌سازی نهایی) از لایه خروجی مدل هستند. تعداد این مقادیر برابر با `num_classes` (در اینجا ۳) است.
    
-   **`Predicted Probabilities (Softmax)`:** این‌ها احتمالات پیش‌بینی شده برای هر یک از کلاس‌ها هستند. مجموع آن‌ها برای هر نمونه (در اینجا، برای تنها یک نمونه) برابر با ۱ خواهد بود. در مثال بالا:
    
    -   احتمال `کاهش`: 24.56%
        
    -   احتمال `بدون تغییر`: 50.09%
        
    -   احتمال `افزایش`: 25.35%
        
-   **`Predicted Class Index`:** شاخص کلاسی که بالاترین احتمال را دارد. در این مثال، شاخص `1` بالاترین احتمال را دارد.
    
-   **`Predicted Stock Movement for next day`:** برچسب متناظر با شاخص پیش‌بینی شده، که در اینجا 'بدون تغییر' است.
    

* * *

امیدوارم این مثال‌های نمونه و شبیه‌سازی خروجی، درک بهتری از نحوه کار مدل و ساختار داده‌های ورودی و خروجی به شما داده باشد. این یک گام مهم در مسیر ساخت یک سیستم پیش‌بینی عملی است.

آیا سوالی در مورد نحوه تولید داده‌های نمونه یا شبیه‌سازی خروجی دارید؟



