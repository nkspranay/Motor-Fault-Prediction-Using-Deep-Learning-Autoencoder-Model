"""
train_model.py
===============
Trains the autoencoder on cleaned motor data.
Only trains on stable, normal (non-fault) data.

Usage:
    python train_model.py

Outputs (all in python/):
    model.pth
    scaler.pkl
    threshold.npy
    feature_thresholds.npy
    training_report/   (plots + summary)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import joblib
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ─────────────── CONFIG ───────────────────
WINDOW_SIZE    = 20       # Timesteps per sample
BATCH_SIZE     = 64
MAX_EPOCHS     = 100
LR             = 0.001
PATIENCE       = 10       # Early stopping patience (epochs)
THRESHOLD_PCT  = 95       # Percentile for anomaly threshold (not 3-sigma)
TRAIN_SPLIT    = 0.80

FEATURES = ["voltage", "current", "power", "temperature", "humidity", "vibration"]
N_FEATURES = len(FEATURES)

INPUT_DIM  = WINDOW_SIZE * N_FEATURES   # 20 × 6 = 120

# Paths
#DATA_PATH   = "../data/processed/motor_data_clean.csv"
#REPORT_DIR  = "training_report"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "data", "processed", "motor_data_clean.csv")
REPORT_DIR = os.path.join(BASE_DIR, "training_report")

# ─────────────── MODEL ────────────────────
class AutoEncoder(nn.Module):
    """
    Symmetric autoencoder with bottleneck.
    Input: flattened window (WINDOW_SIZE × N_FEATURES)
    Output: same shape (reconstruction)

    Architecture:
        Encoder: 120 → 64 → 32 → 16  (bottleneck)
        Decoder: 16  → 32 → 64 → 120
    """
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),          # Output in [0,1] to match MinMaxScaler
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────── HELPERS ──────────────────

def print_section(title: str):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Sliding window: returns array of shape (n_windows, window_size * n_features)"""
    windows = [
        data[i : i + window_size].flatten()
        for i in range(len(data) - window_size)
    ]
    return np.array(windows, dtype=np.float32)
    


# ─────────────── LOAD DATA ────────────────

def load_data():
    print_section("1. Loading data")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Clean data not found at {DATA_PATH}\n"
            f"Run clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    print(f"  Total rows      : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")

    # ── Use only stable (non-transition) rows for training ──
    stable = df[~df["is_transition"]].copy()
    print(f"  Stable rows     : {len(stable):,}")

    # ── Use only known speed states (drop 'unknown') ──
    stable = stable[stable["speed_state"] != "unknown"]
    print(f"  After unknown drop : {len(stable):,}")

    if len(stable) < WINDOW_SIZE * 10:
        raise ValueError(
            f"Not enough stable data ({len(stable)} rows). "
            f"Record more sessions."
        )

    return stable


# ─────────────── SCALE & WINDOW ───────────

def prepare_data(df: pd.DataFrame):
    print_section("2. Scaling and windowing")

    X = df[FEATURES].values

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    #joblib.dump(scaler, "scaler.pkl")
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    print(f"  Scaler saved    : scaler.pkl")

    # Create sliding windows
    X_windows = create_windows(X_scaled, WINDOW_SIZE)
    print(f"  Windows created : {len(X_windows):,}")
    print(f"  Window shape    : {X_windows.shape}")

    # Train / validation split (no shuffle — preserves temporal order)
    split      = int(TRAIN_SPLIT * len(X_windows))
    X_train    = X_windows[:split]
    X_val      = X_windows[split:]

    print(f"  Train windows   : {len(X_train):,}")
    print(f"  Val windows     : {len(X_val):,}")

    return X_train, X_val, scaler


# ─────────────── TRAIN ────────────────────

def train(X_train: np.ndarray, X_val: np.ndarray):
    print_section("3. Training")

    train_tensor = torch.tensor(X_train)
    val_tensor   = torch.tensor(X_val)

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model     = AutoEncoder(INPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )

    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_state   = None
    patience_ctr = 0

    print(f"  Max epochs   : {MAX_EPOCHS}")
    print(f"  Patience     : {PATIENCE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Learning rate: {LR}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss   = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_out  = model(val_tensor)
            val_loss = criterion(val_out, val_tensor).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        # ── Early stopping ──
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train: {epoch_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Best: {best_val:.6f} | "
                  f"Patience: {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # ── Restore best weights ──
    model.load_state_dict(best_state)
    #torch.save(model.state_dict(), "model.pth")
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "model.pth"))
    print(f"\n  ✔ Best model saved (val_loss={best_val:.6f})")

    return model, train_losses, val_losses


# ─────────────── THRESHOLDS ───────────────

def compute_thresholds(model: AutoEncoder, X_train: np.ndarray):
    """
    Compute anomaly thresholds from TRAINING data reconstruction errors only.
    Uses percentile instead of 3-sigma to handle skewed error distributions.
    """
    print_section("4. Computing thresholds")

    model.eval()
    train_tensor = torch.tensor(X_train)

    with torch.no_grad():
        recon  = model(train_tensor)
        errors = (recon - train_tensor) ** 2     # shape: (n, INPUT_DIM)

    # ── Overall MSE per window ──
    overall_errors = errors.mean(dim=1).numpy()  # shape: (n,)

    threshold = float(np.percentile(overall_errors, THRESHOLD_PCT))
    warning_threshold = float(np.percentile(overall_errors, 85))
    #np.save("threshold.npy", threshold)
    np.save(os.path.join(BASE_DIR, "threshold.npy"), threshold)
    #np.save("warning_threshold.npy", warning_threshold)
    np.save(os.path.join(BASE_DIR, "warning_threshold.npy"), warning_threshold)

    print(f"  Overall threshold ({THRESHOLD_PCT}th pct) : {threshold:.6f}")
    print(f"  Warning threshold (85th pct)  : {warning_threshold:.6f}")

    # ── Feature-wise thresholds ──
    # Reshape errors to (n_windows, WINDOW_SIZE, N_FEATURES)
    feat_errors = errors.numpy().reshape(-1, WINDOW_SIZE, N_FEATURES)
    feat_errors = feat_errors.mean(axis=1)       # shape: (n, N_FEATURES)

    feature_thresholds = np.percentile(feat_errors, THRESHOLD_PCT, axis=0)
    #np.save("feature_thresholds.npy", feature_thresholds)
    np.save(os.path.join(BASE_DIR, "feature_thresholds.npy"), feature_thresholds)

    print(f"\n  Feature thresholds ({THRESHOLD_PCT}th pct):")
    for name, val in zip(FEATURES, feature_thresholds):
        print(f"    {name:12s} : {val:.6f}")

    return overall_errors, threshold, warning_threshold, feature_thresholds


# ─────────────── PLOTS ────────────────────

def save_report(train_losses, val_losses, overall_errors, threshold, warning_threshold):
    print_section("5. Saving training report")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Loss curves ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss", color="#3b82f6")
    ax.plot(val_losses,   label="Val Loss",   color="#f97316")
    ax.set_title("Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ loss_curves.png")

    # ── Reconstruction error distribution ──
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(overall_errors, bins=100, color="#6366f1", alpha=0.7, label="Recon Error")
    ax.axvline(threshold,         color="red",    linestyle="--", linewidth=1.5,
               label=f"Fault threshold ({THRESHOLD_PCT}th pct) = {threshold:.5f}")
    ax.axvline(warning_threshold, color="orange", linestyle="--", linewidth=1.5,
               label=f"Warning threshold (85th pct) = {warning_threshold:.5f}")
    ax.set_title("Reconstruction Error Distribution (Training Data)")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "error_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ error_distribution.png")

    # ── Training summary text ──
    summary = (
        f"Training Summary\n"
        f"{'─' * 40}\n"
        f"Epochs run        : {len(train_losses)}\n"
        f"Final train loss  : {train_losses[-1]:.6f}\n"
        f"Final val loss    : {val_losses[-1]:.6f}\n"
        f"Best val loss     : {min(val_losses):.6f} @ epoch {val_losses.index(min(val_losses)) + 1}\n"
        f"Fault threshold   : {threshold:.6f} ({THRESHOLD_PCT}th percentile)\n"
        f"Warning threshold : {warning_threshold:.6f} (85th percentile)\n"
        f"Window size       : {WINDOW_SIZE}\n"
        f"Features          : {', '.join(FEATURES)}\n"
        f"Input dim         : {INPUT_DIM}\n"
    )
    path = os.path.join(REPORT_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(summary)
    print(f"  ✔ summary.txt")
    print(f"\n{summary}")


# ─────────────── MAIN ─────────────────────

def main():
    df                = load_data()
    X_train, X_val, _ = prepare_data(df)
    model, t_losses, v_losses = train(X_train, X_val)
    overall_errors, threshold, warning_threshold, _ = compute_thresholds(model, X_train)
    save_report(t_losses, v_losses, overall_errors, threshold, warning_threshold)

    print_section("Done")
    print(f"  Files saved:")
    print(f"    model.pth")
    print(f"    scaler.pkl")
    print(f"    threshold.npy")
    print(f"    warning_threshold.npy")
    print(f"    feature_thresholds.npy")
    print(f"    {REPORT_DIR}/\n")


if __name__ == "__main__":
    main()