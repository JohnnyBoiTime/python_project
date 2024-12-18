import torch
import torch.nn as nn

print(torch.__version__)
print(torch.cuda.is_available())

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, outputSize)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x);