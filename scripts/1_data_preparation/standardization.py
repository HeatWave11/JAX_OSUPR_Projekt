# file: standardize_data.py

import pandas as pd
import nltk
import re
import pathlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# --- NLTK Setup (Corrected Version) ---
# This block checks if the necessary NLTK components are available.
# If not, it downloads them. This is more robust than catching DownloadError.
def download_nltk_data():
    """Checks for and downloads required NLTK data packages."""
    required_packages = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet.zip',
        'stopwords': 'corpora/stopwords.zip',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger.zip',
        'omw-1.4': 'corpora/omw-1.4.zip'
    }

    all_downloaded = True
    for pkg_id, path in required_packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK package '{pkg_id}' not found. Downloading...")
            nltk.download(pkg_id, quiet=True)
            all_downloaded = False

    if all_downloaded:
        print("All required NLTK packages are already available.")

# Run the check/download function
download_nltk_data()

# --- Initialize Global Tools ---
# These are loaded once to be reused by the function, which is efficient.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def standardize_tweet_text(text: str) -> str:
    """
    Cleans and preprocesses a single string of text.

    This function performs the following steps:
    1. Handles non-string inputs (like missing values) gracefully.
    2. Converts text to lowercase.
    3. Removes URLs.
    4. Removes punctuation, special characters, and numbers.
    5. Tokenizes the text into words.
    6. Removes common English stopwords.
    7. Lemmatizes words using Part-of-Speech (POS) tagging for better accuracy.
    8. Joins the processed words back into a single string.
    """
    # 1. Handle potential missing values (NaNs) or other non-string data
    if not isinstance(text, str):
        return ""

    # 2. Convert to lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 4. Remove punctuation, numbers, and special characters. Keep only letters and spaces.
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Tokenize the text into a list of words
    words = word_tokenize(text)

    # 6. & 7. Remove stopwords and Lemmatize with POS tagging
    tagged_words = nltk.pos_tag(words)
    processed_words = []
    for word, tag in tagged_words:
        # Filter out stopwords and single-character words (often noise)
        if word not in stop_words and len(word) > 1:
            # Convert NLTK's POS tag to a format WordNetLemmatizer understands
            if tag.startswith('J'):
                pos = 'a'  # Adjective
            elif tag.startswith('V'):
                pos = 'v'  # Verb
            elif tag.startswith('N'):
                pos = 'n'  # Noun
            elif tag.startswith('R'):
                pos = 'r'  # Adverb
            else:
                pos = 'n'  # Default to noun if tag is not recognized

            processed_words.append(lemmatizer.lemmatize(word, pos))

    # 8. Reconstruct the sentence from processed words
    return ' '.join(processed_words)

# This is the main execution block that runs when you execute the script
if __name__ == "__main__":
    # --- Defining Robust File Paths with pathlib ---
    # Get the path to the directory containing this script (location scripts/)
    script_dir = pathlib.Path(__file__).parent
    # Get the path to the project root (the parent of scripts/)
    project_root = script_dir.parent.parent # .parent twice, because there are two folders between, not only one
    # Define the data directory and output directory paths
    data_dir = project_root / "data"

    print(f"Project Root Directory: {project_root}")
    print(f"Data Directory: {data_dir}")

    print("Starting data standardization process...")


    # Load the raw CSV files that were downloaded from Kaggle
    try:
        raw_train_path = data_dir / "twitter_training.csv"
        raw_valid_path = data_dir / "twitter_validation.csv"
        train_df = pd.read_csv(raw_train_path, header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
        valid_df = pd.read_csv(raw_valid_path, header=None, names=['TweetID', 'Entity', 'Sentiment', 'TweetContent'])
    except FileNotFoundError:
        print(f"ERROR: Make sure your data files exist in the '{data_dir}' directory.")
        exit()

    # Initialize tqdm for pandas so we can see a progress bar
    tqdm.pandas(desc="Standardizing Training Data")
    # Apply our standardization function to the 'TweetContent' column.
    # '.progress_apply' is the same as '.apply' but with a progress bar.
    train_df['ProcessedContent'] = train_df['TweetContent'].progress_apply(standardize_tweet_text)

    tqdm.pandas(desc="Standardizing Validation Data")
    valid_df['ProcessedContent'] = valid_df['TweetContent'].progress_apply(standardize_tweet_text)

    print("\nStandardization complete.")

    # --- Save to the data directory ---
    clean_train_df = train_df[['ProcessedContent', 'Sentiment']]
    clean_valid_df = valid_df[['ProcessedContent', 'Sentiment']]

    # Define the output file paths using our robust data_dir path
    train_output_path = data_dir / "standardized_training_data.csv"
    valid_output_path = data_dir / "standardized_validation_data.csv"

    clean_train_df.to_csv(train_output_path, index=False)
    clean_valid_df.to_csv(valid_output_path, index=False)

    print(f"\nSuccessfully saved clean data to:")
    print(f"- {train_output_path}")
    print(f"- {valid_output_path}")
