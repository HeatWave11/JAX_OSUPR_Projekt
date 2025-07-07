# file: scripts/2_training_classical/train_and_save_jax_implementations.py

# --- Imports ---
import jax
import jax.numpy as jnp
import numpy as np # Use regular numpy for non-JAX tasks like building the vocab
import optax
import pandas as pd
from sklearn.metrics import classification_report
from collections import Counter
import time
import pathlib
import joblib

# ==============================================================================
# PART 1: FROM-SCRATCH TF-IDF VECTORIZER
# ==============================================================================
def build_vocab(corpus, max_features=10000):
    """Builds a vocabulary from a text corpus."""
    all_text = ' '.join(corpus)
    word_counts = Counter(all_text.split())
    vocab = {word: i for i, (word, count) in enumerate(word_counts.most_common(max_features))}
    return vocab

class TfidfVectorizerJAX:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idf_ = None

    def fit(self, corpus):
        """Calculates IDF values from the corpus."""
        num_docs = len(corpus)
        df = np.zeros(self.vocab_size)
        for text in corpus:
            found_words = set(text.split())
            for word in found_words:
                if word in self.vocab:
                    df[self.vocab[word]] += 1

        self.idf_ = jnp.log((num_docs + 1) / (df + 1)) + 1
        return self

    def transform(self, corpus):
        """Transforms a corpus into a TF-IDF matrix."""
        num_docs = len(corpus)
        tf = np.zeros((num_docs, self.vocab_size))
        for i, text in enumerate(corpus):
            doc_word_counts = Counter(text.split())
            for word, count in doc_word_counts.items():
                if word in self.vocab:
                    tf[i, self.vocab[word]] = count

        row_sums = tf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tf_normalized = tf / row_sums

        tfidf_matrix = jax.device_put(tf_normalized) * self.idf_
        return tfidf_matrix

# ==============================================================================
# PART 2: FROM-SCRATCH NAIVE BAYES (JAX FUNCTIONAL STYLE)
# ==============================================================================
def fit_naive_bayes(X, y, alpha=1.0):
    """Calculates the parameters for a Naive Bayes model."""
    n_samples, n_features = X.shape
    unique_classes, class_counts = jnp.unique(y, return_counts=True)
    n_classes = len(unique_classes)

    class_log_prior = jnp.log(class_counts / n_samples)
    feature_counts = jnp.array([X[y == c].sum(axis=0) for c in unique_classes])
    total_counts_per_class = feature_counts.sum(axis=1, keepdims=True)
    feature_log_prob = jnp.log(feature_counts + alpha) - jnp.log(total_counts_per_class + alpha * n_features)

    return {'class_log_prior': class_log_prior, 'feature_log_prob': feature_log_prob}

@jax.jit
def predict_naive_bayes(params, X):
    """Makes predictions using the trained Naive Bayes parameters."""
    log_probas = (X @ params['feature_log_prob'].T) + params['class_log_prior']
    return jnp.argmax(log_probas, axis=1)

# ==============================================================================
# PART 3: FROM-SCRATCH LOGISTIC REGRESSION (JAX FUNCTIONAL STYLE)
# ==============================================================================
def train_logistic_regression(X_train, y_train, n_classes, n_features, key):
    """Trains a Logistic Regression model using JAX and Optax."""
    # Initialize parameters
    W = jax.random.normal(key, (n_features, n_classes)) * 0.01
    b = jnp.zeros(n_classes)
    params = {'W': W, 'b': b}

    # Define the learning rate and optimizer
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(params)

    # Define the loss function
    def loss_fn(params, X, y):
        logits = X @ params['W'] + params['b']
        one_hot_y = jax.nn.one_hot(y, n_classes)
        return -jnp.mean(jnp.sum(one_hot_y * jax.nn.log_softmax(logits), axis=1))

    # JIT-compile the training step
    @jax.jit
    def train_step(params, opt_state, X_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    print("\nTraining Logistic Regression with JAX/Optax...")
    for epoch in range(10): # 10 epochs
        epoch_loss = 0
        perms = jax.random.permutation(key, len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[perms], y_train[perms]

        for i in range(0, len(X_train), 128): # Batch size 128
            X_batch = X_train_shuffled[i:i+128]
            y_batch = y_train_shuffled[i:i+128]
            params, opt_state, loss = train_step(params, opt_state, X_batch, y_batch)
            epoch_loss += loss

        avg_loss = epoch_loss / (len(X_train) // 128)
        print(f"Epoch {epoch+1}/10, Avg Loss: {avg_loss:.4f}")

    return params

@jax.jit
def predict_logistic_regression(params, X):
    logits = X @ params['W'] + params['b']
    return jnp.argmax(logits, axis=1)

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Setup ---
    key = jax.random.PRNGKey(0)
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models" / "jax_from_scratch"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load and Prepare Data ---
    print("Loading standardized data...")
    train_df = pd.read_csv(data_dir / "standardized_training_data.csv").dropna()
    valid_df = pd.read_csv(data_dir / "standardized_validation_data.csv").dropna()

    label_map = {label: i for i, label in enumerate(train_df['Sentiment'].unique())}
    label_map_rev = {i: label for label, i in label_map.items()}
    y_train = jnp.array(train_df['Sentiment'].map(label_map).values)
    y_valid = jnp.array(valid_df['Sentiment'].map(label_map).values)

    # --- 3. TF-IDF Vectorization ---
    print("\nVectorizing with from-scratch TF-IDF...")
    start_time = time.time()
    vocab = build_vocab(train_df['ProcessedContent'])
    vectorizer = TfidfVectorizerJAX(vocab)
    vectorizer.fit(train_df['ProcessedContent'])
    X_train_tfidf = vectorizer.transform(train_df['ProcessedContent'])
    X_valid_tfidf = vectorizer.transform(valid_df['ProcessedContent'])
    print(f"Vectorization took {time.time() - start_time:.2f}s")

    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer_jax.joblib")
    print("TF-IDF Vectorizer (JAX version) saved.")

    # --- 4. Naive Bayes Training and Evaluation ---
    print("\nTraining Naive Bayes (JAX functional style)...")
    start_time = time.time()
    nb_params = fit_naive_bayes(X_train_tfidf, y_train)
    print(f"Training took {time.time() - start_time:.2f}s")

    predictions_nb = predict_naive_bayes(nb_params, X_valid_tfidf)
    print("\nClassification Report (JAX Naive Bayes):")
    print(classification_report(y_valid, predictions_nb, target_names=label_map_rev.values()))

    joblib.dump(nb_params, models_dir / "naive_bayes_params_jax.joblib")
    print("Naive Bayes parameters (JAX version) saved.")

    # --- 5. Logistic Regression Training and Evaluation ---
    n_classes = len(label_map)
    n_features = X_train_tfidf.shape[1]

    start_time = time.time()
    lr_params = train_logistic_regression(X_train_tfidf, y_train, n_classes, n_features, key)
    print(f"Logistic Regression training took {time.time() - start_time:.2f}s")

    predictions_lr = predict_logistic_regression(lr_params, X_valid_tfidf)
    print("\nClassification Report (JAX Logistic Regression):")
    print(classification_report(y_valid, predictions_lr, target_names=label_map_rev.values()))

    joblib.dump(lr_params, models_dir / "logistic_regression_params_jax.joblib")
    print("Logistic Regression parameters (JAX version) saved.")