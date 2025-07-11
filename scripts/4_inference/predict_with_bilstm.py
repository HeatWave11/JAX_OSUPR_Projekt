# file: scripts/4_inference/predict_with_bilstm.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pathlib
import joblib

# ==============================================================================
### Step 1: Define the necessary functions and classes ###
# To make this script independent, we copy the definitions for our model
# and our text processing functions. In a final application, these would
# live in the `src` folder and be imported.
# ==============================================================================

# --- Copied from standardize_data.py ---
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# We can assume NLTK data is downloaded from previous steps.
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
class BiLSTMClassifier(nn.Module):
    """The Bidirectional LSTM model architecture (copied)."""
    vocab_size: int
    num_classes: int
    embedding_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(x)

        forward_lstm = nn.RNN(nn.LSTMCell(features=self.hidden_dim), return_carry=True)
        forward_carry_out, _ = forward_lstm(x)
        forward_final_state = forward_carry_out[0]

        x_reversed = jnp.flip(x, axis=1)
        backward_lstm = nn.RNN(nn.LSTMCell(features=self.hidden_dim), return_carry=True)
        backward_carry_out, _ = backward_lstm(x_reversed)
        backward_final_state = backward_carry_out[0]

        x = jnp.concatenate([forward_final_state, backward_final_state], axis=-1)
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

# ==============================================================================
### Step 2: The Prediction Pipeline (Corrected) ###
# ==============================================================================

# First, define the pure Python function.
def _predict_internal(params, model_input, model_instance):
    """Internal prediction logic before JIT compilation."""
    logits = model_instance.apply({'params': params}, model_input, train=False)
    return jnp.argmax(logits, -1)

# Second, create the JIT-compiled version of the function.
# This is where we correctly tell JAX that 'model_instance' is static.
predict = jax.jit(_predict_internal, static_argnames='model_instance')

def get_sentiment_prediction(text_input: str, artifacts: dict):
    """
    Takes a raw text string and returns the model's sentiment prediction.
    This function orchestrates the entire process.
    """
    # Unpack the necessary artifacts
    model = artifacts['model']
    params = artifacts['params']
    vocab = artifacts['vocab']
    label_map_rev = artifacts['label_map_rev']
    max_len = artifacts['max_len']

    # 1. Standardize the raw input text
    standardized_text = standardize_tweet_text(text_input)

    # 2. Convert the standardized text to a numerical sequence and pad it.
    sequence = text_to_sequence(standardized_text, vocab, max_len)

    # 3. Format for the model
    model_input = jnp.expand_dims(jnp.array(sequence), axis=0)

    # 4. Get the prediction
    # This now calls the JIT-compiled 'predict' function correctly.
    pred_index_array = predict(params=params, model_input=model_input, model_instance=model)
    pred_index = pred_index_array[0]

    # 5. Convert the predicted index back to a human-readable label.
    return label_map_rev[int(pred_index)]


# ==============================================================================
### Step 3: Main execution block for the interactive test ###
# ==============================================================================
if __name__ == "__main__":
    print("Setting up the BiLSTM predictor...")

    # --- Define paths and constants ---
    # This MUST match the value from your training script.
    MAX_LEN_FROM_TRAINING = 60

    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent.parent
    models_dir = project_root / "models" / "lstm_flax" # The folder where you saved the BiLSTM artifacts

    # --- Load all the saved artifacts ---
    print("Loading model artifacts...")
    # Use jax.tree_util.tree_map to convert loaded NumPy arrays back to JAX arrays
    from jax.tree_util import tree_map
    numpy_params = joblib.load(models_dir / "lstm_params.joblib")
    params = tree_map(jnp.array, numpy_params)

    vocab = joblib.load(models_dir / "lstm_vocab.joblib")
    label_map = joblib.load(models_dir / "lstm_label_map.joblib")
    label_map_rev = {v: k for k, v in label_map.items()}

    # --- Recreate the model structure ---
    model = BiLSTMClassifier(
        vocab_size=len(vocab) + 2, # +2 for PAD and OOV tokens
        num_classes=len(label_map)
    )

    # --- Bundle everything into a single dictionary for convenience ---
    artifacts = {
        'model': model,
        'params': params,
        'vocab': vocab,
        'label_map_rev': label_map_rev,
        'max_len': MAX_LEN_FROM_TRAINING
    }

    print("✅ Predictor is ready. Type your text below or 'quit' to exit.")

    # --- Interactive Loop ---
    while True:
        user_text = input("> ")
        if user_text.lower() == 'quit':
            break

        if not user_text.strip():
            continue

        prediction = get_sentiment_prediction(user_text, artifacts)
        print(f"  -> Prediction: {prediction}\n")