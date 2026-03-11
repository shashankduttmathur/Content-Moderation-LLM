from collections import Counter
import re

TOKEN_PATTERN = re.compile(r"\b\w+\b")

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

class Vocab:
    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        most_common = counter.most_common(self.max_size - len(self.itos))
        for token, _ in most_common:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def encode(self, text, max_len):
        tokens = tokenize(text)
        ids = [self.stoi.get(t, 1) for t in tokens][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids
