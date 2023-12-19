import torch.nn as nn


class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden layer
            output_size: Number of output classes
        """
        super(CustomNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        return out
