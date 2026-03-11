import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from config import *
from vocab import Vocab
from dataset import ModerationDataset
from model import ModerationLLM

def evaluate():
    vocab = Vocab(MAX_VOCAB_SIZE)
    val_ds = ModerationDataset("data/val.csv", vocab)
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModerationLLM(MAX_VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("saved_models/moderation_model.pt", map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(p)
            trues.extend(y.tolist())

    print(classification_report(trues, preds))

if __name__ == "__main__":
    evaluate()
