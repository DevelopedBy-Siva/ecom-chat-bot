import random
import json
import torch
from intent_classifier_model import CustomNeuralNetwork
from text_processing_utils import create_bag_of_words, tokenize_sentence

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intent data
with open('data.json', 'r') as json_data:
    intents_data = json.load(json_data)

# pre-trained intent & load model
MODEL_FILE = "intent_model.pth"
model_data = torch.load(MODEL_FILE)

# Extract model parameters and data
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data['all_words']
tags = model_data['intent_tags']
model_state = model_data["model_state"]

# Initialize the custom neural model
model = CustomNeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Bot name
bot_name = "Jarvis"

# Start the chat loop
print("Let's chat! (type 'quit' to exit)")
while True:
    # Get user input
    user_input = input("You: ")
    if user_input == "quit":
        break

    # Process the user input
    user_input_tokens = tokenize_sentence(user_input)
    bag_of_words = create_bag_of_words(user_input_tokens, all_words)
    bag_of_words = bag_of_words.reshape(1, bag_of_words.shape[0])
    bag_of_words = torch.from_numpy(bag_of_words).to(device)

    # Make a prediction using the model
    output = model(bag_of_words)
    _, predicted = torch.max(output, dim=1)

    # Predicted intent tag
    predicted_tag = tags[predicted.item()]

    # check confidence level
    probabilities = torch.softmax(output, dim=1)
    confidence = probabilities[0][predicted.item()]

    if confidence.item() > 0.75:
        for intent in intents_data['intents']:
            if predicted_tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
