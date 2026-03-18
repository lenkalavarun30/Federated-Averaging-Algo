import torch
from torch import optim
from backend.model import CNNModel

class Client:
    def __init__(self, data_loader):
        self.model = CNNModel()
        self.data_loader = data_loader

    def train(self, epochs=1):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.model.state_dict()
