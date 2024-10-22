import warnings
warnings.filterwarnings('ignore')
# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.preprocess import remove_stopwords, clean_text, stemming

app = Flask(__name__)

# Load the model and preprocessor
def load_model():
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocess the input text
def preprocess_input(text):
    text = remove_stopwords(text)
    text = clean_text(text)
    text = stemming(text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request
        data = request.get_json(force=True)
        input_text = data['text']

        # Load the model
        model = load_model()

        # Preprocess the input text
        processed_text = preprocess_input(input_text)

        # Make prediction
        prediction = model.predict([processed_text])

        labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        # Prepare the response
        result = {label: int(pred) for label, pred in zip(labels, prediction[0])}
        
        return jsonify(result)

    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        raise CustomException(e)

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    try:
        # Get the JSON request which contains a list of comments
        data = request.get_json(force=True)
        comments = data['comments']

        # Load the model
        model = load_model()

        # Preprocess each comment
        processed_comments = [preprocess_input(comment) for comment in comments]

        # Make predictions in bulk
        predictions = model.predict(processed_comments)

        labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

        # Prepare the response for all comments
        result = []
        for idx, pred in enumerate(predictions):
            prediction_result = {label: int(p) for label, p in zip(labels, pred)}
            result.append({"comment": comments[idx], "prediction": prediction_result})
        
        return jsonify(result)

    except Exception as e:
        logging.error("Error in bulk prediction: %s", str(e))
        raise CustomException(e)

if __name__ == '__main__':
    app.run(debug=True)
