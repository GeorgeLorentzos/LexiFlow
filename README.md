# LexiFlow

**LexiFlow** is a lightweight next-word predictor built with PyTorch. It uses an LSTM-based language model to predict the most likely next word in a sentence, making it useful for text generation and language modeling experiments.

## Features

- Predicts the next word given a sequence of words.
- LSTM-based model with embeddings and fully connected layers.
- Configurable embedding and hidden sizes based on vocabulary size.
- Greedy word prediction function for simple inference.
- Easy to train on custom text datasets.

## How It Works

1. **Data Preparation:** The input text is tokenized into words and converted to integer indices for training sequences.
2. **Model Architecture:** 
   - **Embedding layer:** Converts word indices to dense vectors.
   - **LSTM layer:** Captures contextual information from sequences.
   - **MLP layer:** Maps LSTM outputs to the vocabulary space for predictions.
3. **Training:** Uses `CrossEntropyLoss` and `AdamW` optimizer.
4. **Prediction:** The `predict()` function generates the next word(s) based on input text.

## Example Usage

```python
# Predict the next word(s)
print(predict("hello how are"))
# Output might be: "you"
