import streamlit as st
import subprocess
import re
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
st.set_page_config(
    page_title="Federated Learning Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡 Federated Learning Attack Detection Dashboard")
st.subheader("Backdoor Attack Simulation")

st.markdown("Dataset: **Chest X-Ray Pneumonia Dataset**")
st.markdown("Clients: **4** | Rounds: **3** | Attack: **Backdoor**")

st.divider()

start = st.button("Start Training")

# containers
log_box = st.empty()

col1, col2 = st.columns(2)

norm_box = col1.empty()
cosine_box = col2.empty()

attack_box = st.empty()

# data
logs = []
norms = {}
cosines = {}
attacks = []

def parse_line(line):

    logs.append(line)

    # norm
    m = re.search(r"Client (\d+) update norm: ([0-9\.]+)", line)

    if m:
        cid = int(m.group(1))
        val = float(m.group(2))
        norms[cid] = val

    # cosine
    m = re.search(r"Client (\d+) cosine similarity: ([0-9\.]+)", line)

    if m:
        cid = int(m.group(1))
        val = float(m.group(2))
        cosines[cid] = val

    # attacks
    if "ATTACK DETECTED" in line or "Malicious Client Detected" in line:
        attacks.append(line)


if start:

    st.info("Initializing training... loading dataset and models")

    cmd = [
        "python",
        "experiments/federated_train.py",
        "backdoor",
        "--rounds",
        "3",
        "--num_clients",
        "4"
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
            "\n".join(logs[-120:]),
            height=350
        )

        # norms
        if norms:

            df = pd.DataFrame({
                "Client": list(norms.keys()),
                "Update Norm": list(norms.values())
            })

            norm_box.subheader("Client Update Norms")
            norm_box.dataframe(df, use_container_width=True)

        # cosine
        if cosines:

            df = pd.DataFrame({
                "Client": list(cosines.keys()),
                "Cosine Similarity": list(cosines.values())
            })

            cosine_box.subheader("Cosine Similarity")
            cosine_box.dataframe(df, use_container_width=True)

        # attack detection
        attack_box.subheader("Attack Detection")

        if attacks:

            for a in attacks[-5:]:
                attack_box.error(a)

        else:
            attack_box.success("No attacks detected yet")

st.divider()

st.markdown("### Project Summary")

st.write("""
This project demonstrates **attack detection in Federated Learning systems**.

Multiple hospitals collaboratively train a **CNN pneumonia detection model**
without sharing their data.

A malicious client performs a **Backdoor Attack** by poisoning its model updates.

The server detects malicious clients using:

• **Norm-based anomaly detection**  
• **Cosine similarity detection**

The malicious client is identified when its update deviates from the majority of clients.
""")

st.caption("Secure Federated Learning Research Prototype")