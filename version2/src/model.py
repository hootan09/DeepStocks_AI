import torch, torch.nn as nn

class StockNet(nn.Module):
    def __init__(self, num_price_feat, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(num_price_feat, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.news_fc = nn.Sequential(nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 2)
        )
    def forward(self, x_num, x_news):
        _, (h_n, _) = self.lstm(x_num)
        h_num = h_n[-1]
        h_news = self.news_fc(x_news)
        return self.classifier(torch.cat([h_num, h_news], dim=1))

class EarlyStopper:
    def __init__(self, patience, path):
        self.best = float("inf"); self.patience = patience; self.counter = 0; self.path = path
    def __call__(self, val_loss, model):
        if val_loss < self.best:
            self.best, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
        return self.counter >= self.patience
