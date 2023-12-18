from flask import Flask
from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def home():
    return "Hello"


if __name__ == "__main__":
    app.run()
