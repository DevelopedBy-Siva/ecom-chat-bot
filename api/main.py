from chatbot.bot import Bot
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

bot = Bot()


@app.route("/chat")
def home():
    query = request.args.get('query')
    if not query:
        raise ValueError("Query cannot be empty.")
    return bot.display()


if __name__ == "__main__":
    app.run()
