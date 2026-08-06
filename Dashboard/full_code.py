import streamlit as st
import subprocess
import re
import sys

st.set_page_config(
    page_title="Secure Federated Learning",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡 Secure Federated Learning Dashboard")
st.write("Federated Learning Attack Detection System")

st.divider()

# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.header("Training Settings")

    attack_type = st.selectbox(
        "Select Attack",
        ["none", "label_flipping", "model_poisoning", "backdoor"]
    )

    rounds = st.slider("Rounds", 1, 5, 3)

    clients = st.slider("Clients", 2, 6, 4)

    start = st.button("Start Training")


# -----------------------
# Containers
# -----------------------

log_box = st.empty()

col1, col2 = st.columns(2)

norm_box = col1.empty()
cosine_box = col2.empty()

attack_box = st.empty()

# -----------------------
# Storage
# -----------------------

logs = []
norms = {}
cosines = {}
attacks = []

# -----------------------
# Parser
# -----------------------

def parse_line(line):

    logs.append(line)

    # Norm
    m = re.search(r"Client (\d+) update norm: ([0-9\.]+)", line)

    if m:
        cid = int(m.group(1))
        val = float(m.group(2))
        norms[cid] = val

    # Cosine
    m = re.search(r"Client (\d+) cosine similarity: ([0-9\.]+)", line)

    if m:
        cid = int(m.group(1))
        val = float(m.group(2))
        cosines[cid] = val

    # Attack detection
    if "ATTACK DETECTED" in line or "Malicious Client Detected" in line:
        attacks.append(line)


# -----------------------
# Start Training
# -----------------------

if start:

    st.info("Initializing training... loading dataset and models")

    cmd = [
        sys.executable,
        "-m",
        "experiments.federated_train",
        attack_type,
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

        line = line.strip()

        parse_line(line)

        # logs
        log_box.text_area(
            "Training Logs",
            "\n".join(logs[-100:]),
            height=350
        )

        # norms
        if norms:

            norm_box.subheader("Client Update Norms")

            norm_box.table({
                "Client": list(norms.keys()),
                "Norm": list(norms.values())
            })

        # cosine
        if cosines:

            cosine_box.subheader("Cosine Similarity")

            cosine_box.table({
                "Client": list(cosines.keys()),
                "Cosine": list(cosines.values())
            })

        # attacks
        attack_box.subheader("Attack Detection")

        if attacks:

            for a in attacks[-10:]:
                attack_box.warning(a)

        else:

            attack_box.success("No attacks detected yet")

st.divider()

st.caption("Secure Federated Learning Research Demo")