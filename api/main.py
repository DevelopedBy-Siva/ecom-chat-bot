from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import nltk_sentiment_analyzer.sentiment as sentiment_analyzer

load_dotenv()


app = Flask(__name__)

host = os.getenv("CLIENT_URL")
CORS(app, resources={r"/*": {"origins": host}})


@app.route("/")
def home():
    return sentiment_analyzer.run()


if __name__ == "__main__":
    app.run()
