# train_model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import logging
from src.exception import CustomException
import sys
from src.components import preprocess

# Load preprocessor (TF-IDF) and preprocessed data
def load_preprocessed_data():
    with open('artifacts/preprocessor.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    
    df = pd.read_csv("dataset/preprocessed_data.csv")
    X = df['comment_text']
    y = df.drop(columns=['comment_text'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Preprocessed Data Loaded")
    return tfidf, X_train, X_test, y_train, y_test

# Function to run model training pipeline
def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    logging.info("Pipeline Running")
    X_train = X_train.fillna('')
    X_test = X_test.fillna('')
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    pred_probs = pipeline.predict_proba(X_test)

    print("roc_auc_score: ", roc_auc_score(y_test, pred_probs))
    print("accuracy: ", accuracy_score(y_test, predictions))
    print("classification_report:")
    print(classification_report(y_test, predictions, target_names=y_train.columns))

# Train the model and save as .pkl
def train_model():
    try:
        logging.info("Training Started")
        tfidf, X_train, X_test, y_train, y_test = load_preprocessed_data()

        # Choose the classifier (Logistic Regression in this case)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1)),
        ])

        # Train and evaluate the model
        run_pipeline(pipeline, X_train, X_test, y_train, y_test)
        logging.info("Pipeline Run Successful")
        # Save the trained model to .pkl file
        with open('artifacts/model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info("Model Dumped")
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    train_model()
