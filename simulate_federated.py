
import torch
import numpy as np

from core.federated.federated_agent import FederatedAgent
from core.federated.coordinator import Coordinator
from core.federated.autoencoder_model import AutoencoderModel
from core.persistence.model_store import ModelStore


# ----------------------------
# Synthetic Data Generator
# ----------------------------
def generate_normal_data(samples=300):
    return np.random.normal(loc=0.0, scale=1.0, size=(samples, 20))


def generate_attack_data(samples=50):
    return np.random.normal(loc=4.0, scale=1.5, size=(samples, 20))


# ----------------------------
# Federated Training
# ----------------------------
def train_federated(rounds=5):

    input_dim = 20
    store = ModelStore()

    # Initialize global model
    global_model = AutoencoderModel(input_dim=input_dim)

    # Load existing model if present
    if store.load(global_model):
        print("📂 Existing global model loaded.")
        print("📂 Resuming from saved model.")
    else:
        print("🆕 Training new global model.")

    coordinator = Coordinator()

    for r in range(rounds):

        print(f"\n🚀 Federated Round {r+1}")

        # Create agents
        agents = [
            FederatedAgent(generate_normal_data(), input_dim),
            FederatedAgent(generate_normal_data(), input_dim),
            FederatedAgent(generate_normal_data(), input_dim)
        ]

        # Local training
        for agent in agents:
            agent.local_train()

        # Collect payloads
        payloads = [agent.get_payload() for agent in agents]

        # Byzantine-aware aggregation (UPDATED LINE INCLUDED)
        global_weights = coordinator.aggregate(payloads, global_model)

        # Update global model
        global_model.load_state_dict(global_weights)

        print("✔ Aggregated from 3 valid clients.")

    # Save trained model
    store.save(global_model)
    print("💾 Global model saved to disk.")

    return global_model


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model):

    normal_data = generate_normal_data()
    attack_data = generate_attack_data()

    normal_errors = model.compute_reconstruction_error(normal_data)
    attack_errors = model.compute_reconstruction_error(attack_data)

    threshold = np.percentile(normal_errors, 99)

    TP = np.sum(attack_errors > threshold)
    FN = np.sum(attack_errors <= threshold)
    FP = np.sum(normal_errors > threshold)
    TN = np.sum(normal_errors <= threshold)

    print("\n🔎 Detection Evaluation:")
    print("Threshold:", round(threshold, 6))
    print(f"TP: {TP}  FP: {FP}")
    print(f"TN: {TN}  FN: {FN}")

    fpr = FP / (FP + TN)
    recall = TP / (TP + FN)

    print("False Positive Rate:", round(fpr, 4))
    print("Detection Recall:", round(recall, 4))


# ----------------------------
# Main
# ----------------------------
def main():
    print("🏭 Federated Edge Security — Production Mode")
    model = train_federated(rounds=5)
    evaluate(model)


if __name__ == "__main__":
    main()
 

# ---------------------------
# My intro : I am Bhoopendra Singh Bhadauria, student of  Central University of Haryana, Mahendragarh. I am persuing B.Tech Computer Science and Engineering.
# ---------------------------
