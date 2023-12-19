import random
import json
import torch
from textProcessingUtils import create_bag_of_words, tokenize_sentence
from intentClassifierModel import CustomNeuralNetwork


class ChatBot:

    def __init__(self):
        # Bot name
        self.bot_name = "Jarvis"

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load intent data
        with open('data/data.json', 'r') as json_data:
            self.intents_data = json.load(json_data)

            # pre-trained intent & load model
            MODEL_FILE = "data/intent_model.pth"
            model_data = torch.load(MODEL_FILE)

            # Extract model parameters and data
            input_size = model_data["input_size"]
            hidden_size = model_data["hidden_size"]
            output_size = model_data["output_size"]
            self.all_words = model_data['all_words']
            self.tags = model_data['intent_tags']
            model_state = model_data["model_state"]

            # Initialize the custom neural model
            self.model = CustomNeuralNetwork(input_size, hidden_size, output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()

    def generate_chat(self, query, is_logged_in):
        # Process the user input
        user_input_tokens = tokenize_sentence(query)
        bag_of_words = create_bag_of_words(user_input_tokens, self.all_words)
        bag_of_words = bag_of_words.reshape(1, bag_of_words.shape[0])
        bag_of_words = torch.from_numpy(bag_of_words).to(self.device)

        # Make a prediction using the model
        output = self.model(bag_of_words)
        _, predicted = torch.max(output, dim=1)

        # Predicted intent tag
        predicted_tag = self.tags[predicted.item()]

        # check confidence level
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted.item()]

        # Default response
        response = "Could you please explain this in a different way?"

        if confidence.item() > 0.75:
            for intent in self.intents_data['intents']:
                if predicted_tag == intent["tag"]:
                    response = random.choice(intent['responses'])
        return {
            "bot_name": self.bot_name,
            "query": response
        }
