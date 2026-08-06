import torch

def model_poisoning_attack(weights):

    print(" Model Poisoning Attack Executed")

    poisoned = {}

    for key, value in weights.items():

        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            noise = torch.randn_like(value) * 0.2
            poisoned[key] = value + noise
        else:
            poisoned[key] = value

    return poisoned