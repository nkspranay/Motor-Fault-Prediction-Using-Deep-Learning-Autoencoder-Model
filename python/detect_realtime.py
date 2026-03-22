import torch
import serial
import numpy as np
import joblib
from torch import nn
from collections import deque

# ---------------- CONFIG ----------------
WINDOW_SIZE = 20
CONFIRMATION_COUNT = 3

# ---------------- MODEL ----------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW_SIZE * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, WINDOW_SIZE * 5)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = AutoEncoder()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ---------------- LOAD FILES ----------------
scaler = joblib.load("scaler.pkl")
THRESHOLD = float(np.load("threshold.npy"))
feature_thresholds = np.load("feature_thresholds.npy")

feature_names = ["Voltage", "Current", "Power", "Temperature", "Vibration"]

# Get scaling limits (for inverse transform)
feature_mins = scaler.data_min_
feature_maxs = scaler.data_max_

# ---------------- SERIAL ----------------
ser = serial.Serial("COM5", 115200, timeout=1)

# ---------------- BUFFER ----------------
buffer = deque(maxlen=WINDOW_SIZE)
anomaly_counter = 0

print("Real-time anomaly detection started...")
print("Overall Threshold:", THRESHOLD)
print("Feature Thresholds:", feature_thresholds)

# ---------------- LOOP ----------------
while True:
    try:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        vals = raw.split(",")
        if len(vals) != 5:
            continue

        vals = np.array([float(v) for v in vals])

        # Scale input
        vals_scaled = scaler.transform([vals])[0]

        buffer.append(vals_scaled)

        # Wait until buffer fills
        if len(buffer) < WINDOW_SIZE:
            continue

        # Prepare input
        window = np.array(buffer).reshape(1, -1)
        x = torch.tensor(window, dtype=torch.float32)

        with torch.no_grad():
            recon = model(x)

            # -------- Overall Loss --------
            loss = torch.mean((recon - x) ** 2).item()

            # -------- Feature-wise Error --------
            error = (recon - x).numpy().reshape(WINDOW_SIZE, 5)
            feature_errors = error.mean(axis=0)

        # -------- Anomaly Logic --------
        if loss > THRESHOLD:
            anomaly_counter += 1
        else:
            anomaly_counter = 0

        # -------- OUTPUT --------
        if anomaly_counter >= CONFIRMATION_COUNT:
            print(f"\n⚠ CONFIRMED ANOMALY | MSE={loss:.6f}")

            faults = []

            for i, fe in enumerate(feature_errors):
                if fe > feature_thresholds[i]:

                    # Last value from buffer (scaled)
                    current_scaled = buffer[-1][i]

                    # Convert to original value
                    current_value = current_scaled * (feature_maxs[i] - feature_mins[i]) + feature_mins[i]

                    # Determine type of fault
                    if current_value > feature_maxs[i]:
                        status = "HIGH"
                    elif current_value < feature_mins[i]:
                        status = "LOW"
                    else:
                        status = "ABNORMAL"

                    faults.append(
                        f"{feature_names[i]} ({status}) | Value={current_value:.2f}"
                    )

            if faults:
                print("Fault Details:")
                for f in faults:
                    print("  →", f)
            else:
                print("⚠ General anomaly (no specific feature identified)")

        else:
            print(f"Normal | MSE={loss:.6f}")

    except ValueError:
        continue
    except KeyboardInterrupt:
        print("\nStopped by user.")
        break