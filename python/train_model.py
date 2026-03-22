import torch
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 20
BATCH_SIZE = 32
EPOCHS = 50

# ---------------- Load CSV ----------------
data = pd.read_csv("../data/motor_data.csv")

# ---------------- Data Cleaning ----------------
data = data.dropna()
data = data[(data["V"] > 0) & (data["I"] >= 0)]

if len(data) < WINDOW_SIZE:
    raise ValueError("Not enough data for window size. Collect more data.")

# ---------------- Extract Features ----------------
X = data[["V", "I", "P", "T", "VIB"]].values

# ---------------- Scaling ----------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# ---------------- Sliding Windows ----------------
def create_windows(data, window_size):
    return np.array([data[i:i+window_size] for i in range(len(data)-window_size)])

X_windows = create_windows(X, WINDOW_SIZE)

# Flatten
X_windows = X_windows.reshape(len(X_windows), -1)

# ---------------- Train / Validation Split ----------------
split = int(0.8 * len(X_windows))
X_train = X_windows[:split]
X_val = X_windows[split:]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
                          batch_size=BATCH_SIZE, shuffle=True)

val_tensor = torch.tensor(X_val, dtype=torch.float32)

# ---------------- Model ----------------
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

model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- Training ----------------
print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch in train_loader:
        x_batch = batch[0]

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, x_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(val_tensor)
        val_loss = criterion(val_output, val_tensor).item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

torch.save(model.state_dict(), "model.pth")
print(" Model saved")

# ---------------- Threshold Calculation ----------------
model.eval()
with torch.no_grad():
    recon = model(torch.tensor(X_windows, dtype=torch.float32))

    # -------- Overall error --------
    total_errors = torch.mean((recon - torch.tensor(X_windows, dtype=torch.float32)) ** 2, dim=1).numpy()

    # -------- Feature-wise error --------
    errors = (recon - torch.tensor(X_windows, dtype=torch.float32)) ** 2
    errors = errors.numpy().reshape(-1, WINDOW_SIZE, 5)

    feature_errors = errors.mean(axis=1)

# -------- Overall Threshold --------
threshold = total_errors.mean() + 3 * total_errors.std()
np.save("threshold.npy", threshold)

# -------- Feature-wise Thresholds --------
feature_thresholds = feature_errors.mean(axis=0) + 3 * feature_errors.std(axis=0)
np.save("feature_thresholds.npy", feature_thresholds)

print("\nThresholds:")
print(f"Overall     : {threshold:.6f}")
print(f"Voltage     : {feature_thresholds[0]:.6f}")
print(f"Current     : {feature_thresholds[1]:.6f}")
print(f"Power       : {feature_thresholds[2]:.6f}")
print(f"Temperature : {feature_thresholds[3]:.6f}")
print(f"Vibration   : {feature_thresholds[4]:.6f}")