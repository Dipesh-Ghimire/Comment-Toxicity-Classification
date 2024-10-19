import pickle
from src.components.preprocess import clean_text,stemming
from src.exception import CustomException
from src.logger import logging
import sys

# Load preprocessor and trained model
with open('artifacts/preprocessor.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_comment(comment):
    try:
        # Preprocess the comment (use similar functions from preprocess.py)
        comment = clean_text(comment)
        comment = stemming(comment)
        
        # Predict using the loaded model
        prediction = model.predict([comment])
        logging.info("prediction:")
        return prediction
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    # Example comment
    comment = "i hate you you're the most annoying person."
    logging.info(comment)
    prediction = predict_comment(comment)
    print(prediction)
