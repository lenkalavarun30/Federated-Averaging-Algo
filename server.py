class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_weights):
        avg_weights = {}
        for key in client_weights[0].keys():
            avg_weights[key] = sum([cw[key] for cw in client_weights]) / len(client_weights)
        self.global_model.load_state_dict(avg_weights)
        return self.global_model.state_dict()
