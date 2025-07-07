# file: scripts/2_training_classical/train_and_save_classical_models.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import time
import pathlib
import joblib

def train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid, model_name="Model"):
    """ A helper function to train a model and print its evaluation. """
    print("-" * 50)
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    predictions = model.predict(X_valid)
    print("\nClassification Report:")
    print(classification_report(y_valid, predictions))
    print("-" * 50)
    # Return the trained model
    return model

if __name__ == "__main__":
    # --- 1. Define Paths (remember to go up two levels!) ---
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models" / "classical"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models will be saved to: {models_dir}")

    # --- 2. Load Standardized Data ---
    print("\nLoading standardized data...")
    train_df = pd.read_csv(data_dir / "standardized_training_data.csv").dropna()
    valid_df = pd.read_csv(data_dir / "standardized_validation_data.csv").dropna()

    # --- 3. Vectorize Text with TF-IDF ---
    print("\nVectorizing text...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['ProcessedContent'])
    X_valid_tfidf = tfidf_vectorizer.transform(valid_df['ProcessedContent'])

    # Save the fitted vectorizer immediately, as it's needed for all models
    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")

    # --- 4. Train, Evaluate, and Save Models ---
    y_train = train_df['Sentiment']
    y_valid = valid_df['Sentiment']

    # --- Naive Bayes ---
    nb_model_trained = train_and_evaluate_model(
        MultinomialNB(), X_train_tfidf, y_train, X_valid_tfidf, y_valid, "Naive Bayes"
    )
    joblib.dump(nb_model_trained, models_dir / "naive_bayes_model.joblib")
    print(f"Naive Bayes model saved.")

    # --- Logistic Regression ---
    lr_model_trained = train_and_evaluate_model(
        LogisticRegression(max_iter=1000, random_state=42, solver='saga'),
        X_train_tfidf, y_train, X_valid_tfidf, y_valid, "Logistic Regression"
    )
    joblib.dump(lr_model_trained, models_dir / "logistic_regression_model.joblib")
    print(f"Logistic Regression model saved.")

    print("\nAll classical models trained and saved successfully!")