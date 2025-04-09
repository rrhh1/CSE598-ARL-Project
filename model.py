import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(30, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x