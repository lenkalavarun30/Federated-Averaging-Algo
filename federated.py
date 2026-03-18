from backend.client import Client
from backend.server import Server
from backend.model import CNNModel
def federated_training(clients, rounds=3):
    global_model = CNNModel()
    server = Server(global_model)

    for r in range(rounds):
        client_weights = []
        for client in clients:
            weights = client.train(epochs=1)
            client_weights.append(weights)

        global_weights = server.aggregate(client_weights)

        for client in clients:
            client.model.load_state_dict(global_weights)

    return global_model
