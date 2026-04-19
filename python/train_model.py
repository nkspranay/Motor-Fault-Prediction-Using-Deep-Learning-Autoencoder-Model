"""
train_model.py
===============
Trains the autoencoder on cleaned motor data.
Only trains on stable, normal (non-fault) data.

Key fixes vs naive approach:
  - THRESHOLD_PCT = 99.9  (was 99.5) — only extreme tail triggers FAULT
  - WARNING_PCT   = 99.0  (was 97.0) — stops normal noise from being "warning"
  - IQR-based outlier rejection before fitting the scaler, so corrupt sensor
    glitches (e.g. a 0.99V voltage spike seen in real logs) don't shift the
    scaler's min/max and inflate MSE on normal data
  - Scaler fitted with SCALER_MARGIN headroom so real-time values slightly
    outside the training range don't hard-clip to 0/1 and spike MSE

Usage:
    python train_model.py                 # trains both states
    python train_model.py --state running
    python train_model.py --state off
"""

import os
import argparse
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
WINDOW_SIZE   = 20
BATCH_SIZE    = 64
MAX_EPOCHS    = 100
LR            = 0.001
PATIENCE      = 10

# Raised significantly vs old 99.5/97.0 values.
# 99.9/99.0 means only the very extreme tail of *training* errors triggers a
# flag — real-time normal data has a wide comfortable band to live in.
THRESHOLD_PCT = 99.9
WARNING_PCT   = 99.0

TRAIN_SPLIT   = 0.80

# Expand the scaler's min/max by this fraction on each side.
# If training voltage range is 220-240 V, scaler effectively treats 209-252 V
# as [0,1] — so a 241V real reading doesn't hard-clip to 1.0 and spike MSE.
SCALER_MARGIN = 0.05

FEATURES   = ["voltage", "current", "power", "temperature", "humidity", "vibration"]
N_FEATURES = len(FEATURES)
INPUT_DIM  = WINDOW_SIZE * N_FEATURES   # 20 × 6 = 120

VALID_STATES = ("running", "off")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "motor_data_clean.csv")


# ─────────────── MODEL ────────────────────
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),        nn.ReLU(),
            nn.Linear(32, 64),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────── HELPERS ──────────────────
def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.array(
        [data[i : i + window_size].flatten()
         for i in range(len(data) - window_size + 1)],
        dtype=np.float32,
    )


def iqr_clean(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Remove rows where any feature is beyond 3×IQR from Q1/Q3.
    Prevents corrupt sensor glitches from skewing the scaler's range.
    """
    mask = pd.Series(True, index=df.index)
    for col in features:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        mask   &= df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)
    removed = (~mask).sum()
    if removed > 0:
        print(f"  IQR outlier removal  : dropped {removed} rows "
              f"({removed / len(df) * 100:.2f}%)")
    return df[mask]


def fit_scaler_with_margin(X: np.ndarray, margin: float) -> MinMaxScaler:
    """
    Fit MinMaxScaler then expand its effective range by `margin` on each side.
    Real-time values slightly outside training range won't hard-clip to 0/1.
    """
    scaler = MinMaxScaler()
    scaler.fit(X)
    headroom          = (scaler.data_max_ - scaler.data_min_) * margin
    scaler.data_min_  = scaler.data_min_  - headroom
    scaler.data_max_  = scaler.data_max_  + headroom
    scaler.data_range_= scaler.data_max_  - scaler.data_min_
    scaler.scale_     = 1.0 / scaler.data_range_
    scaler.min_       = -scaler.data_min_ * scaler.scale_
    return scaler


# ─────────────── LOAD DATA ────────────────
def load_data(state: str) -> pd.DataFrame:
    print_section(f"1. Loading data  [state={state}]")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Clean data not found at {DATA_PATH}\nRun clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    missing = [c for c in FEATURES + ["is_transition", "speed_state"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print(f"  Total rows           : {len(df):,}")
    stable = df[~df["is_transition"] & (df["speed_state"] == state)].copy()
    print(f"  Stable '{state}' rows    : {len(stable):,}")

    if len(stable) < WINDOW_SIZE * 10:
        raise ValueError(
            f"Not enough stable '{state}' data ({len(stable)} rows). "
            "Record more sessions."
        )

    print(f"\n  Raw feature statistics:")
    for col in FEATURES:
        s = stable[col]
        print(f"    {col:12s}  min={s.min():.3f}  max={s.max():.3f}  "
              f"mean={s.mean():.3f}  std={s.std():.3f}")

    return stable


# ─────────────── SCALE & WINDOW ───────────
def prepare_data(df: pd.DataFrame, out_dir: str):
    print_section("2. Cleaning, scaling and windowing")

    df_clean = iqr_clean(df, FEATURES)
    X = df_clean[FEATURES].values

    feature_means = df_clean[FEATURES].mean().values
    feature_stds  = df_clean[FEATURES].std().values
    np.save(os.path.join(out_dir, "feature_means.npy"), feature_means)
    np.save(os.path.join(out_dir, "feature_stds.npy"),  feature_stds)
    print(f"  Feature means : {np.round(feature_means, 3)}")
    print(f"  Feature stds  : {np.round(feature_stds,  3)}")

    scaler   = fit_scaler_with_margin(X, SCALER_MARGIN)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"\n  Scaler saved  (margin={SCALER_MARGIN*100:.0f}% headroom)")
    print(f"  Effective min : {np.round(scaler.data_min_, 2)}")
    print(f"  Effective max : {np.round(scaler.data_max_, 2)}")

    X_scaled  = np.clip(scaler.transform(X), 0.0, 1.0)
    X_windows = create_windows(X_scaled, WINDOW_SIZE)
    print(f"\n  Windows       : {len(X_windows):,}  shape={X_windows.shape}")

    split   = int(TRAIN_SPLIT * len(X_windows))
    X_train = X_windows[:split]
    X_val   = X_windows[split:]
    print(f"  Train         : {len(X_train):,}")
    print(f"  Val           : {len(X_val):,}")

    return X_train, X_val, scaler


# ─────────────── TRAIN ────────────────────
def train(X_train: np.ndarray, X_val: np.ndarray, out_dir: str):
    print_section("3. Training")

    train_tensor = torch.tensor(X_train)
    val_tensor   = torch.tensor(X_val)
    loader       = DataLoader(TensorDataset(train_tensor),
                              batch_size=BATCH_SIZE, shuffle=True)

    model     = AutoEncoder(INPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_losses, val_losses = [], []
    best_val, best_sd = float("inf"), None
    patience_ctr      = 0

    print(f"  Max epochs : {MAX_EPOCHS}  Patience : {PATIENCE}  LR : {LR}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()

        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train={epoch_loss:.6f}  Val={val_loss:.6f}  "
                  f"Best={best_val:.6f}  Patience={patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_sd)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    print(f"\n  ✔ Best model saved (val_loss={best_val:.6f})")
    return model, train_losses, val_losses


# ─────────────── THRESHOLDS ───────────────
def compute_thresholds(model: AutoEncoder, X_train: np.ndarray, out_dir: str):
    print_section("4. Computing thresholds")

    model.eval()
    with torch.no_grad():
        t    = torch.tensor(X_train)
        errs = ((model(t) - t) ** 2).mean(dim=1).numpy()

    print("  Error distribution across training windows:")
    for pct in [50, 75, 90, 95, 97, 99, 99.5, 99.9]:
        print(f"    {pct:5.1f}th pct : {np.percentile(errs, pct):.6f}")

    threshold = float(np.percentile(errs, THRESHOLD_PCT))
    warn_thr  = float(np.percentile(errs, WARNING_PCT))

    np.save(os.path.join(out_dir, "threshold.npy"),         threshold)
    np.save(os.path.join(out_dir, "warning_threshold.npy"), warn_thr)

    print(f"\n  ✔ Fault threshold   ({THRESHOLD_PCT}th pct) : {threshold:.6f}")
    print(f"  ✔ Warning threshold ({WARNING_PCT}th pct)   : {warn_thr:.6f}")
    print(f"  ✔ Ratio fault/warn  : {threshold/warn_thr:.2f}x")

    return errs, threshold, warn_thr


# ─────────────── REPORT ───────────────────
def save_report(train_losses, val_losses, errors, threshold, warn_thr,
                out_dir: str, state: str):
    print_section("5. Saving training report")
    report_dir = os.path.join(out_dir, "training_report")
    os.makedirs(report_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train", color="#3b82f6")
    ax.plot(val_losses,   label="Val",   color="#f97316")
    ax.set_title(f"Loss curves [{state}]")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "loss_curves.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors, bins=100, color="#6366f1", alpha=0.7)
    ax.axvline(threshold, color="red",    linestyle="--", linewidth=1.5,
               label=f"Fault  ({THRESHOLD_PCT}th) = {threshold:.5f}")
    ax.axvline(warn_thr,  color="orange", linestyle="--", linewidth=1.5,
               label=f"Warning ({WARNING_PCT}th) = {warn_thr:.5f}")
    ax.set_title(f"Reconstruction Error Distribution [{state}]")
    ax.set_xlabel("MSE"); ax.set_ylabel("Count"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "error_distribution.png"), dpi=150)
    plt.close()

    summary = (
        f"Training Summary [{state}]\n{'─'*45}\n"
        f"Epochs run        : {len(train_losses)}\n"
        f"Best val loss     : {min(val_losses):.6f} @ "
        f"epoch {val_losses.index(min(val_losses))+1}\n"
        f"Fault threshold   : {threshold:.6f} ({THRESHOLD_PCT}th pct)\n"
        f"Warning threshold : {warn_thr:.6f} ({WARNING_PCT}th pct)\n"
        f"Scaler margin     : {SCALER_MARGIN*100:.0f}%\n"
        f"Window size       : {WINDOW_SIZE}\n"
        f"Features          : {', '.join(FEATURES)}\n"
    )
    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print("  ✔ Plots and summary.txt saved")
    print(f"\n{summary}")


# ─────────────── MAIN ─────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=VALID_STATES, default=None)
    args   = parser.parse_args()
    states = VALID_STATES if args.state is None else (args.state,)

    for state in states:
        out_dir = os.path.join(BASE_DIR, "models", state)
        os.makedirs(out_dir, exist_ok=True)
        df                        = load_data(state)
        X_train, X_val, _         = prepare_data(df, out_dir)
        model, t_losses, v_losses = train(X_train, X_val, out_dir)
        errors, thr, warn_thr     = compute_thresholds(model, X_train, out_dir)
        save_report(t_losses, v_losses, errors, thr, warn_thr, out_dir, state)
        print_section(f"Done — {state}")
        print(f"  Output dir : {out_dir}/\n")


if __name__ == "__main__":
    main()