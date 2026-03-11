import torch
from config import *
from vocab import Vocab
from model import ModerationLLM

def predict(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab(MAX_VOCAB_SIZE)
    model = ModerationLLM(MAX_VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("saved_models/moderation_model.pt", map_location=device))
    model.eval()

    x = torch.tensor([vocab.encode(text, MAX_LEN)]).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return ID2LABEL[pred]

if __name__ == "__main__":
    print(predict("I will hurt you"))
