## Model Architecture
The system uses a lightweight Transformer encoder to understand text context and classify content into moderation categories.

Text → Tokenization

Tokens → Embeddings

Transformer Encoder (Self‑Attention)

Pooling Layer

Linear Classifier → Moderation Label


# PyTorch Content Moderation (Mini‑LLM Classifier)

Minimal Transformer-based text classifier for content moderation.

## Setup
pip install -r requirements.txt

## Train
python src/train.py

## Evaluate
python src/evaluate.py

## Predict
python src/predict.py

## Author
Shashank Dutt