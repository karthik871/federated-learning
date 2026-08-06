import sys
import random
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_pneumonia import PneumoniaModel
from clients.client import Client
from data.pneumonia_loader import get_data_loaders

from attacks.label_flipping import label_flipping_attack
from attacks.model_poisoning import model_poisoning_attack
from attacks.backdoor_attack import backdoor_attack


# -----------------------------
# Command line arguments
# -----------------------------

attack = "none"
rounds = 3
num_clients = 4

if len(sys.argv) > 1:
    attack = sys.argv[1]

if "--rounds" in sys.argv:
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1])

if "--num_clients" in sys.argv:
    num_clients = int(sys.argv[sys.argv.index("--num_clients") + 1])

print("Starting Federated Learning Training")
print("Attack Mode:", attack)
print("Rounds:", rounds)
print("Clients:", num_clients)


# -----------------------------
# Detection Algorithm 1
# Norm based detection
# -----------------------------

def detect_by_norm(client_updates):

    norms = []

    for update in client_updates:

        norm = 0

        for key in update:

            value = update[key]

            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                norm += torch.norm(value).item()

        norms.append(norm)

    avg_norm = sum(norms) / len(norms)

    suspicious_clients = []

    for i, n in enumerate(norms):

        print(f"Client {i+1} update norm: {n:.4f}")

        if n > avg_norm * 1.5:
            print(f"ATTACK DETECTED (Norm): Client {i+1}")
            suspicious_clients.append(i)

    return suspicious_clients


# -----------------------------
# Detection Algorithm 2
# Cosine similarity detection
# -----------------------------

def detect_by_cosine(client_updates):

    vectors = []

    for update in client_updates:

        flat_params = []

        for key in update:

            value = update[key]

            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                flat_params.append(value.flatten())

        vectors.append(torch.cat(flat_params))

    reference = torch.median(torch.stack(vectors), dim=0).values

    suspicious_clients = []

    for i, vec in enumerate(vectors):

        similarity = F.cosine_similarity(vec, reference, dim=0)

        print(f"Client {i+1} cosine similarity: {similarity:.4f}")

        if similarity < 0.85:
            print(f"ATTACK DETECTED (Cosine): Client {i+1}")
            suspicious_clients.append(i)

    return suspicious_clients


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load dataset
# -----------------------------

train_loader, test_loader = get_data_loaders()


# -----------------------------
# Initialize global model
# -----------------------------

global_model = PneumoniaModel().to(device)


# -----------------------------
# Create clients
# -----------------------------

clients = []

for i in range(num_clients):

    client_model = PneumoniaModel().to(device)

    client = Client(client_model, train_loader, device)

    clients.append(client)


# -----------------------------
# Select malicious client
# -----------------------------

malicious_client = random.randint(0, num_clients - 1)

print("Malicious client selected: Client", malicious_client + 1)


# -----------------------------
# Federated training loop
# -----------------------------

for round_num in range(rounds):

    print("\nFederated Round", round_num + 1)

    client_updates = []

    for i, client in enumerate(clients):

        print("Client", i + 1, "training...")

        client.model.load_state_dict(global_model.state_dict())

        weights = client.train(epochs=1)

        # -----------------------------
        # Apply attack if malicious
        # -----------------------------

        if i == malicious_client:

            print(f"Malicious Client Performing Attack: Client {i+1}")

            if attack == "label_flipping":

                print("Running Label Flipping Attack")

                weights = label_flipping_attack(weights)

            elif attack == "model_poisoning":

                print("Running Model Poisoning Attack")

                weights = model_poisoning_attack(weights)

            elif attack == "backdoor":

                print("Running Backdoor Attack")

                weights = backdoor_attack(weights)

        client_updates.append(weights)


    # -----------------------------
    # Detection
    # -----------------------------

    print("\nRunning Detection Algorithms...\n")

    norm_detected = detect_by_norm(client_updates)

    cosine_detected = detect_by_cosine(client_updates)

    detected_clients = set(norm_detected) | set(cosine_detected)

    if detected_clients:

        for c in detected_clients:
            print(f"Malicious Client Detected by Server: Client {c+1}")

    else:

        print("No malicious clients detected")


    # -----------------------------
    # Aggregation
    # -----------------------------

    new_state_dict = {}

    for key in global_model.state_dict().keys():

        tensors = []

        for update in client_updates:

            value = update[key]

            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                tensors.append(value)

        if tensors:
            new_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        else:
            new_state_dict[key] = global_model.state_dict()[key]

    global_model.load_state_dict(new_state_dict)

    print("Aggregation complete")


print("\nTraining Finished")