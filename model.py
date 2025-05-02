import torch
import torch.nn as nn

from torchvision import models, transforms

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
    

class RNN_Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = 3

        self.rnn = nn.RNN(
            input_size=3,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity='relu',
            bidirectional=False,
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_size, 1)
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.rnn(x, h0)
        y = self.fc(x[:, -1, :])
        return y


class Modified_ResNet(nn.Module):
    def __init__(self):
        super(Modified_ResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 101)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of the last few layers for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        
        self.in_fc = nn.Linear(10, 50176)

    def forward(self, x):
        x = x.transpose(1, 2).to(x.device)
        x = self.in_fc(x)
        
        x = x.view(-1, 3 * 224 * 224)
        
        min_val = torch.min(x, dim=1)[0].view(-1, 1)
        max_val = torch.max(x, dim=1)[0].view(-1, 1)
        x = (x - min_val) / (max_val - min_val)
        
        x = x.view(-1, 3, 224, 224).to(x.device)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        x = self.resnet(x)
        
        return x