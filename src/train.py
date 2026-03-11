import torch
from torch.utils.data import DataLoader
import pandas as pd

from config import *
from vocab import Vocab
from dataset import ModerationDataset
from model import ModerationLLM

def main():
    df = pd.read_csv("data/train.csv")
    texts = df["text"].tolist()

    vocab = Vocab(MAX_VOCAB_SIZE)
    vocab.build(texts)

    train_ds = ModerationDataset("data/train.csv", vocab)
    val_ds = ModerationDataset("data/val.csv", vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModerationLLM(len(vocab.itos)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "saved_models/moderation_model.pt")

if __name__ == "__main__":
    main()
