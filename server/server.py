import torch

class Server:

    def aggregate(self, client_weights):

        avg_weights = {}

        for key in client_weights[0].keys():

            avg_weights[key] = torch.zeros_like(client_weights[0][key])

            for weights in client_weights:
                avg_weights[key] += weights[key]

            avg_weights[key] = avg_weights[key] / len(client_weights)

        return avg_weights