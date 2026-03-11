MAX_VOCAB_SIZE = 20000
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
N_CLASSES = 6

LABEL_MAP = {
    "safe": 0,
    "spam": 1,
    "abusive": 2,
    "hate": 3,
    "nsfw": 4,
    "toxic": 5
}

ID2LABEL = {v:k for k,v in LABEL_MAP.items()}
