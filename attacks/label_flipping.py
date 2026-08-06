def label_flipping_attack(weights):
    print("⚠ Label Flipping Attack executed")

    attacked_weights = {}

    for key in weights.keys():
        attacked_weights[key] = -weights[key]

    return attacked_weights