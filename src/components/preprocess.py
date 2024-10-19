# preprocess.py
import re
import sys
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from src.logger import logging
from src.exception import CustomException

# Download NLTK stopwords if not already
nltk.download('stopwords')

# Define stopwords and stemmer
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Preprocessing functions
def remove_stopwords(text):
    return " ".join([w for w in text.split() if not w in stopwords])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Keep only alphabetic characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()

def stemming(sentence):
    return " ".join([stemmer.stem(word) for word in sentence.split()])

# Preprocess function
def preprocess(df):
    logging.info("Preprocessing Started")
    df['comment_text'] = df['comment_text'].apply(lambda x: remove_stopwords(x))
    df['comment_text'] = df['comment_text'].apply(lambda x: clean_text(x))
    df['comment_text'] = df['comment_text'].apply(lambda x: stemming(x))
    return df

# Load dataset and preprocess
def main():
    try:
        df = pd.read_csv("dataset/train.csv")
        df = df.drop(columns=["id"], axis=1)
        logging.info("Import Train Dataset")
        # Preprocess the comments
        df = preprocess(df)
        df.to_csv("dataset/preprocessed_data.csv",index=False,header=True)
        logging.info("Preprocessing Completed")

        # Split data into training and test sets
        X = df['comment_text']
        y = df.drop(columns=['comment_text'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Train-Test-Split")
        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')

        # Fit on training data
        X_train_tfidf = tfidf.fit_transform(X_train)

        # Save the preprocessor (TF-IDF) as a .pkl file
        with open('artifacts/preprocessor.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        logging.info("Dumped Preprocessor Pickel File")
        return X_train_tfidf, X_test, y_train, y_test
    
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()
