# predict.py
import pickle
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.preprocess import remove_stopwords, clean_text, stemming  # Import existing functions

# Load the entire pipeline from model.pkl
def load_model():
    try:
        with open('artifacts/model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        logging.info("Model Loaded Successfully")
        return pipeline
    except Exception as e:
        raise CustomException(e, sys)

# Preprocess the input text
def preprocess_input(text):
    try:
        # Apply preprocessing steps
        text = remove_stopwords(text)
        text = clean_text(text)
        text = stemming(text)
        logging.info("Text Preprocessing Completed")
        return text
    except Exception as e:
        raise CustomException(e, sys)

# Prediction function for a single text input
def predict_text(text):
    try:
        # Load the trained model (pipeline)
        pipeline = load_model()

        # Preprocess the input text
        processed_text = preprocess_input(text)

        # Make prediction using the pipeline
        prediction = pipeline.predict([processed_text])
        return prediction

    except Exception as e:
        raise CustomException(e, sys)

# Main function to take input from command line and predict
if __name__ == "__main__":
    try:
        # Take the input text to predict from the command line
        input_text = "i hate you"

        # Predict the toxicity labels
        prediction = predict_text(input_text)

        print(prediction)
    except Exception as e:
        raise CustomException(e, sys)