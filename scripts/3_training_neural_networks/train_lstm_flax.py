# file: scripts/3_training_neural_networks/train_lstm_flax.py

# --- Imports ---
import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
import optax

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from collections import Counter
import time
import pathlib
import joblib
from tqdm import tqdm

# ==============================================================================
### PART 1: DATA PREPARATION FOR A SEQUENCE MODEL ###
# We don't use TF-IDF. Instead, we convert text to sequences of integers.
# ==============================================================================

def build_vocab(corpus, max_features=10000):
    """Builds a vocabulary and word-to-index mapping."""
    print("Building vocabulary from training data...")
    word_counts = Counter()
    for text in tqdm(corpus):
        word_counts.update(text.split())

    # Special tokens: 0 for padding, 1 for "out of vocabulary" (OOV) words.
    # We start our actual word mapping from index 2.
    vocab = {word: i+2 for i, (word, count) in enumerate(word_counts.most_common(max_features))}
    vocab_size = len(vocab) + 2 # Add 2 for the PAD and OOV tokens

    print(f"Vocabulary size: {vocab_size} (including PAD and OOV tokens)")
    return vocab, vocab_size

def text_to_sequence(text, vocab, max_len):
    """Converts a single text string to a padded sequence of integers."""
    # Convert words to their integer index, using 1 if the word is not in the vocab (OOV)
    seq = [vocab.get(word, 1) for word in text.split()]

    # Pad or truncate the sequence to max_len
    if len(seq) < max_len:
        # Pad with zeros
        return seq + [0] * (max_len - len(seq))
    else:
        # Truncate from the end
        return seq[:max_len]

# ==============================================================================
### PART 2: A MORE POWERFUL, BIDIRECTIONAL LSTM MODEL (FINAL, ROBUST VERSION) ###
# ==============================================================================

class BiLSTMClassifier(nn.Module):
    """A Bidirectional LSTM model for better performance."""
    vocab_size: int
    num_classes: int
    embedding_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, train: bool):
        # 1. Embedding Layer
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(x)

        # 2. Bidirectional LSTM Layer

        # --- FORWARD PASS ---
        # We define a forward LSTM and run it.
        # The 'carry_out' will be a tuple (hidden_state, cell_state) from the last time step.
        forward_lstm = nn.RNN(nn.LSTMCell(features=self.hidden_dim), return_carry=True)
        forward_carry_out, _ = forward_lstm(x)
        forward_final_state = forward_carry_out[0] # We only need the hidden state

        # --- BACKWARD PASS ---
        # We define a backward LSTM. To make it run backwards, we first reverse the input sequence.
        # The `time_major=False` default means the time axis is at index 1.
        x_reversed = jnp.flip(x, axis=1)
        backward_lstm = nn.RNN(nn.LSTMCell(features=self.hidden_dim), return_carry=True)
        backward_carry_out, _ = backward_lstm(x_reversed)
        backward_final_state = backward_carry_out[0] # Get the final hidden state

        # --- CONCATENATE ---
        # We combine the final states from both directions. This is the essence of a BiLSTM.
        # Shape: (batch_size, 2 * hidden_dim)
        x = jnp.concatenate([forward_final_state, backward_final_state], axis=-1)

        # 3. Dropout Layer
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)

        # 4. Dense Layer
        x = nn.Dense(features=self.num_classes)(x)

        return x
    


# ==============================================================================
### PART 3: TRAINING AND EVALUATION LOGIC (JAX/FLAX PATTERNS) ###
# These are the reusable functions for our training loop.
# ==============================================================================

def create_train_state(model, key, learning_rate, input_shape):
    """Initializes the model and creates a TrainState object."""
    # Initialize the model's parameters (weights)
    params = model.init(key, jnp.ones(input_shape, jnp.int32), train=False)['params']
    # Create an Adam optimizer
    tx = optax.adam(learning_rate)
    # Create a Flax TrainState to hold all the moving parts of training.
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit  # Performance Optimization: JIT-compile this entire function!
def train_step(state, batch, dropout_rng):
    """Performs a single training step (forward pass, loss, backward pass, update)."""
    # Unpack the batch
    inputs, labels = batch

    # Create a new dropout key for this step to ensure randomness
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    # Define the loss function
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            inputs,
            train=True,
            rngs={'dropout': dropout_rng}
        )
        one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss

    # Calculate gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Update model parameters
    state = state.apply_gradients(grads=grads)

    return state, loss, new_dropout_rng

@jax.jit  # Performance Optimization: JIT-compile evaluation too!
def eval_step(state, batch):
    """Performs an evaluation step on a batch of data."""
    inputs, labels = batch
    logits = state.apply_fn({'params': state.params}, inputs, train=False)
    loss = optax.softmax_cross_entropy(jax.nn.one_hot(labels, logits.shape[-1]), logits).mean()
    acc = (jnp.argmax(logits, -1) == labels).mean()
    return loss, acc

# ==============================================================================
### MAIN EXECUTION BLOCK ###
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Setup and Hyperparameters ---
    MAX_LEN = 60         # Maximum length of a tweet sequence
    MAX_FEATURES = 15000 # Maximum number of words in our vocabulary
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 4

    key = jax.random.PRNGKey(0)

    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models" / "lstm_flax"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load and Prepare Data ---
    train_df = pd.read_csv(data_dir / "standardized_training_data.csv").dropna()
    valid_df = pd.read_csv(data_dir / "standardized_validation_data.csv").dropna()

    vocab, vocab_size = build_vocab(train_df['ProcessedContent'], max_features=MAX_FEATURES)

    X_train = np.array([text_to_sequence(text, vocab, MAX_LEN) for text in tqdm(train_df['ProcessedContent'], desc="Processing train data")])
    X_valid = np.array([text_to_sequence(text, vocab, MAX_LEN) for text in tqdm(valid_df['ProcessedContent'], desc="Processing valid data")])

    label_map = {label: i for i, label in enumerate(train_df['Sentiment'].unique())}
    label_map_rev = {i: label for label, i in label_map.items()}
    y_train = np.array(train_df['Sentiment'].map(label_map).values)
    y_valid = np.array(valid_df['Sentiment'].map(label_map).values)

    # --- 3. Initialize Model and TrainState ---
    model = BiLSTMClassifier(vocab_size=vocab_size, num_classes=len(label_map))

    # Create the TrainState object
    key, init_key = jax.random.split(key)
    input_shape = (BATCH_SIZE, MAX_LEN)
    state = create_train_state(model, init_key, LEARNING_RATE, input_shape)

    # --- 4. The Training Loop ---
    print("\nStarting LSTM Training...")
    key, dropout_key = jax.random.split(key)

    for epoch in range(NUM_EPOCHS):
        # Shuffle the training data for each epoch
        key, perm_key = jax.random.split(key)
        perms = jax.random.permutation(perm_key, len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[perms], y_train[perms]

        # Training phase
        with tqdm(range(0, len(X_train), BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]") as pbar:
            for i in pbar:
                batch_X = X_train_shuffled[i:i+BATCH_SIZE]
                batch_y = y_train_shuffled[i:i+BATCH_SIZE]

                state, loss, dropout_key = train_step(state, (batch_X, batch_y), dropout_key)
                pbar.set_postfix(loss=f'{loss:.4f}')

        # Evaluation phase
        val_losses, val_accs = [], []
        for i in tqdm(range(0, len(X_valid), BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
            batch_X = X_valid[i:i+BATCH_SIZE]
            batch_y = y_valid[i:i+BATCH_SIZE]
            loss, acc = eval_step(state, (batch_X, batch_y))
            val_losses.append(loss)
            val_accs.append(acc)

        print(f"Epoch {epoch+1} | Validation Loss: {np.mean(val_losses):.4f} | Validation Accuracy: {np.mean(val_accs):.4f}")

    # --- 5. Final Evaluation and Saving ---
    print("\nTraining complete. Final evaluation on the full validation set...")

    # Predict on the entire validation set
    all_preds = []
    for i in range(0, len(X_valid), BATCH_SIZE):
        batch_X = X_valid[i:i+BATCH_SIZE]
        logits = state.apply_fn({'params': state.params}, batch_X, train=False)
        all_preds.extend(jnp.argmax(logits, -1))

    y_pred = np.array(all_preds)
    y_true_labels = [label_map_rev[l] for l in y_valid]
    y_pred_labels = [label_map_rev[l] for l in y_pred]

    print("\nClassification Report (Flax LSTM):")
    print(classification_report(y_true_labels, y_pred_labels))

    # Save the trained model parameters and the vocabulary
    print("Converting model parameters to NumPy arrays for saving...")
    # Use jax.device_get to convert the entire parameter tree to NumPy arrays
    numpy_params = jax.device_get(state.params)
    joblib.dump(numpy_params, models_dir / "lstm_params.joblib")

    # The rest of the saving code is fine
    joblib.dump(vocab, models_dir / "lstm_vocab.joblib")
    joblib.dump(label_map, models_dir / "lstm_label_map.joblib")
    print(f"Final model artifacts saved to: {models_dir}")