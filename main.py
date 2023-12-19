from chatbot import ChatBot
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

chatbot = ChatBot()


@app.route("/")
def hello_world():
    return "<p>Welcome</p>"


@app.route("/chat", methods=['POST'])
def home():
    try:
        data = request.get_json()

        # Validate request data
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        is_logged_in = data.get('isLoggedIn')
        if not is_logged_in:
            is_logged_in = False

        # Generate Chat
        return chatbot.generate_chat(message, is_logged_in)
    except Exception:
        return jsonify({"error": "Something went wrong"}), 500


if __name__ == "__main__":
    from waitress import serve

    serve(app)
