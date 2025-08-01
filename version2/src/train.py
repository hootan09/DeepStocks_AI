import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot, corr, loss_tot = 0, 0, 0.0
    for x_num, x_news, y in loader:
        x_num, x_news, y = x_num.to(device), x_news.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_num, x_news)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_tot += loss.item()
        preds = out.argmax(1)
        tot += y.size(0); corr += (preds == y).sum().item()
    return loss_tot/len(loader), corr/tot

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels, loss_tot = [], [], 0.0
    for x_num, x_news, y in loader:
        x_num, x_news, y = x_num.to(device), x_news.to(device), y.to(device)
        out = model(x_num, x_news)
        loss_tot += criterion(out, y).item()
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    acc = sum(p==l for p,l in zip(preds, labels))/len(labels)
    return loss_tot/len(loader), acc, preds, labels
