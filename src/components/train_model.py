# train_model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import sys

from src.logger import logging
from src.exception import CustomException

# Load preprocessor (TF-IDF) and preprocessed data
def load_preprocessed_data():
    try:
        # Load the preprocessed dataset
        df = pd.read_csv("dataset/preprocessed_data.csv")
        X = df['comment_text']
        y = df.drop(columns=['comment_text'], axis=1)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Train-Test Split Successful")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)

# Function to run model training pipeline
def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    try:
        logging.info("Pipeline Training Started")
        X_train = X_train.fillna('')  # Fill NaN values
        X_test = X_test.fillna('')

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Get predictions and probabilities
        predictions = pipeline.predict(X_test)
        pred_probs = pipeline.predict_proba(X_test)

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, pred_probs)
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions, target_names=y_train.columns)

        # Print out the metrics
        print(f"roc_auc_score: {roc_auc}")
        print(f"accuracy: {accuracy}")
        print(f"classification_report:\n{class_report}")
        logging.info("Model Evaluation Completed")
    except Exception as e:
        raise CustomException(e, sys)

# Train the model and save as .pkl
def train_model():
    try:
        logging.info("Training Process Started")

        # Load data
        X_train, X_test, y_train, y_test = load_preprocessed_data()

        # Load the preprocessor (TF-IDF)
        with open('artifacts/preprocessor.pkl', 'rb') as f:
            tfidf = pickle.load(f)

        # Create the pipeline
        pipeline = Pipeline([
            ('tfidf', tfidf),  # Use the loaded preprocessor
            ('model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1)),
        ])

        # Run the pipeline
        run_pipeline(pipeline, X_train, X_test, y_train, y_test)

        # Save the trained model
        with open('artifacts/model.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info("Model Saved Successfully")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train_model()
