import torch
from torch.utils.data import Dataset
import pandas as pd
from config import LABEL_MAP, MAX_LEN

class ModerationDataset(Dataset):
    def __init__(self, csv_path, vocab):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = [LABEL_MAP[l] for l in df["label"]]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.vocab.encode(self.texts[idx], MAX_LEN)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)
