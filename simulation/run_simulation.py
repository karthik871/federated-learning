import subprocess

attacks = ["none", "label_flipping", "model_poisoning", "backdoor"]

rounds = 3
clients = 4

print("Starting Federated Learning Attack Simulation\n")

for attack in attacks:

    print("\n====================================")
    print("Running Attack:", attack)
    print("====================================\n")

    cmd = [
        "python",
        "-m",
        "experiments.federated_train",
        attack,
        "--rounds",
        str(rounds),
        "--num_clients",
        str(clients)
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line.strip())

print("\nSimulation Complete")