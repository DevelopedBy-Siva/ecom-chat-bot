import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from textProcessingUtils import create_bag_of_words, tokenize_sentence, stem
from intentClassifierModel import CustomNeuralNetwork

# Load data from JSON file
with open('data/data.json', 'r') as json_file:
    intent_data = json.load(json_file)

# Extract patterns and tags from intent data
all_words = []
intent_tags = []
patterns_tags_pairs = []

for intent in intent_data['intents']:
    tag = intent['tag']
    intent_tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_words = tokenize_sentence(pattern)
        all_words.extend(tokenized_words)
        patterns_tags_pairs.append((tokenized_words, tag))

# Preprocess words
ignore_words = ['?', '.', '!']
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
intent_tags = sorted(set(intent_tags))

# Create training data
training_data_X = []
training_data_y = []

for (pattern_sentence, intent_tag) in patterns_tags_pairs:
    bag_of_words = create_bag_of_words(pattern_sentence, all_words)
    training_data_X.append(bag_of_words)
    label = intent_tags.index(intent_tag)
    training_data_y.append(label)

training_data_X = np.array(training_data_X)
training_data_y = np.array(training_data_y)

# Model parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(training_data_X[0])
hidden_size = 8
output_size = len(intent_tags)
print(input_size, output_size)


# Define custom dataset
class IntentDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.X_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create dataset and dataloader
intent_dataset = IntentDataset(training_data_X, training_data_y)
train_loader = DataLoader(dataset=intent_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize intent classification model
intent_classifier_model = CustomNeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(intent_classifier_model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = intent_classifier_model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save model
model_data = {
    "model_state": intent_classifier_model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "intent_tags": intent_tags
}

file_path = "data/intent_model.pth"
torch.save(model_data, file_path)

print(f'Training complete. Model data saved to {file_path}')
