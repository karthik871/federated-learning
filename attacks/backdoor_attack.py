import torch

def backdoor_attack(weights):

    poisoned = {}

    for key, value in weights.items():

        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            noise = torch.randn_like(value) * 0.1
            poisoned[key] = value + noise
        else:
            poisoned[key] = value

    print("Backdoor Attack Executed")

    return poisoned