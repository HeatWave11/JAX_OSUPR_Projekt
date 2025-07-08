# file: scripts/4_inference/predict_with_lstm.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pathlib
import joblib
from jax.tree_util import tree_map

# --- Step 1: Copy necessary definitions from other scripts ---
# To make this script runnable on its own, we need the model's architecture
# and the text standardization function. In a real application, these would
# live in your `src` folder and be imported.

# --- Copied from standardize_data.py ---
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_data():
    """Checks for and downloads required NLTK data packages."""
    # (Same NLTK download function as before)
    # ...
    pass # Assume it's already downloaded from previous steps

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def standardize_tweet_text(text: str) -> str:
    """Cleans and preprocesses a single string of text (copied)."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    processed_words = []
    for word, tag in tagged_words:
        if word not in stop_words and len(word) > 1:
            pos = 'n'
            if tag.startswith('J'): pos = 'a'
            elif tag.startswith('V'): pos = 'v'
            elif tag.startswith('R'): pos = 'r'
            processed_words.append(lemmatizer.lemmatize(word, pos))
    return ' '.join(processed_words)

# --- Copied from train_lstm_flax.py ---
class LSTMClassifier(nn.Module):
    """A Flax LSTM model using the high-level nn.RNN wrapper (copied)."""
    vocab_size: int
    num_classes: int
    embedding_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(x)
        lstm_cell = nn.LSTMCell(features=self.hidden_dim)
        hidden_states = nn.RNN(lstm_cell)(x)
        x = hidden_states[:, -1, :]
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

def text_to_sequence(text, vocab, max_len):
    """Converts a single text string to a padded sequence of integers (copied)."""
    seq = [vocab.get(word, 1) for word in text.split()] # Use 1 for unknown words
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    else:
        return seq[:max_len]

# --- Step 2: The Prediction Pipeline ---
# This is the core function that ties everything together.
@jax.jit  # JIT-compile the prediction for speed!
def predict(params, model_input):
    """Runs the model forward pass and returns the predicted class index."""
    logits = model.apply({'params': params}, model_input, train=False)
    return jnp.argmax(logits, -1)

def get_sentiment_prediction(text_input: str, model_artifacts: dict):
    """
    Takes a raw text string and returns the model's sentiment prediction.

    Args:
        text_input: The raw string to analyze.
        model_artifacts: A dictionary containing the loaded model, params, vocab, etc.

    Returns:
        A string with the predicted sentiment label.
    """
    # Unpack the artifacts
    model = model_artifacts['model']
    params = model_artifacts['params']
    vocab = model_artifacts['vocab']
    label_map_rev = model_artifacts['label_map_rev']
    max_len = model_artifacts['max_len']

    # 1. Standardize the raw text
    standardized_text = standardize_tweet_text(text_input)

    # 2. Convert to a numerical sequence and pad
    sequence = text_to_sequence(standardized_text, vocab, max_len)

    # 3. Format for the model
    #    - Convert to a JAX array
    #    - Add a batch dimension (the model expects a batch, even if it's just one item)
    model_input = jnp.expand_dims(jnp.array(sequence), axis=0)

    # 4. Get the prediction
    pred_index_array = predict(params, model_input)
    pred_index = pred_index_array[0] # Extract the number from the array

    # 5. Convert the index back to a human-readable label
    return label_map_rev[int(pred_index)]


# --- Step 3: Main execution block ---
if __name__ == "__main__":
    print("Setting up the LSTM predictor...")

    # --- Load all the saved artifacts ---
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent.parent
    models_dir = project_root / "models" / "lstm_flax"

    params = joblib.load(models_dir / "lstm_params.joblib")
    vocab = joblib.load(models_dir / "lstm_vocab.joblib")
    label_map = joblib.load(models_dir / "lstm_label_map.joblib")
    label_map_rev = {v: k for k, v in label_map.items()}

    # --- Recreate the model structure ---
    model = LSTMClassifier(
        vocab_size=len(vocab) + 2, # +2 for PAD and OOV tokens
        num_classes=len(label_map)
    )

    # Bundle everything into a single dictionary for convenience
    artifacts = {
        'model': model,
        'params': params,
        'vocab': vocab,
        'label_map_rev': label_map_rev,
        'max_len': 60 # This MUST match the MAX_LEN from training
    }

    print("Predictor is ready. Type your text below or 'quit' to exit.")

    # --- Interactive Loop ---
    while True:
        user_text = input("> ")
        if user_text.lower() == 'quit':
            break

        prediction = get_sentiment_prediction(user_text, artifacts)
        print(f"  -> Prediction: {prediction}\n")