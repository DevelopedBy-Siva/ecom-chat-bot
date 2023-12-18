from chatbot import Bot
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

bot = Bot()


@app.route("/")
def hello_world():
    return "<p>Welcome</p>"


@app.route("/chat")
def home():
    query = request.args.get('query')
    if not query:
        raise ValueError("Query cannot be empty.")
    return bot.display()


if __name__ == "__main__":
    from waitress import serve
    serve(app)
