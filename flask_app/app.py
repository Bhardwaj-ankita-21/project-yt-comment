from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return comment

# Load the model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load ML model from MLflow registry and vectorizer from local storage."""
    try:
        mlflow.set_tracking_uri("http://ec2-18-208-107-74.compute-1.amazonaws.com:5000/")
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from {model_uri}...")  # Debugging log
        model = mlflow.pyfunc.load_model(model_uri)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully!")  # Debugging log
        return model, vectorizer

    except Exception as e:
        print(f"Error loading model/vectorizer: {e}")
        return None, None

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

if not model or not vectorizer:
    print("❌ Model or vectorizer failed to load. Check MLflow URI and vectorizer path.")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict sentiment from comments."""
    try:
        data = request.json
        if not data or 'comments' not in data:
            return jsonify({"error": "No comments provided"}), 400

        comments = data['comments']
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(response)

    except Exception as e:
        print(f"Prediction failed: {e}")  # Log the error
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode
