from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()


app = Flask(__name__)

host = os.getenv("CLIENT_URL")
CORS(app, resources={r"/*": {"origins": host}})


@app.route("/")
def home():
    return "Hello World"


if __name__ == "__main__":
    app.run()
