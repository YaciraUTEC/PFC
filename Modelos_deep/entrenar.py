import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── Rutas ────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
RESULTS_DIR = _HERE.parent / "gym-pybullet-drones" / "results"
CSV_PATH    = RESULTS_DIR / "datos_CF2X_800ep.csv"
STATS_PATH  = RESULTS_DIR / "stats_normalizacion.json"

# ── Hiperparámetros ──────────────────────────────────────────
WINDOW_SIZE = 50
BATCH_SIZE  = 128
VAL_SPLIT   = 0.2
EPOCHS      = 150
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_COLS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "roll",  "pitch", "yaw",
    "ang_x", "ang_y", "ang_z",
    "target_x", "target_y", "target_z",
    "err_x", "err_y", "err_z",
]
OUTPUT_COLS = ["motor_0", "motor_1", "motor_2", "motor_3"]

print(f"Dispositivo: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Dataset ──────────────────────────────────────────────────
class DroneDataset(Dataset):
    def __init__(self, df, window_size):
        self.samples = []
        for ep in df["episodio"].unique():
            ep_df = df[df["episodio"] == ep].reset_index(drop=True)
            X = ep_df[INPUT_COLS].values.astype(np.float32)
            y = ep_df[OUTPUT_COLS].values.astype(np.float32)
            for i in range(len(X)):
                if i >= window_size:
                    window = X[i - window_size:i]
                else:
                    pad    = np.zeros((window_size - i, X.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, X[:i]]) if i > 0 else np.zeros((window_size, X.shape[1]), dtype=np.float32)
                self.samples.append((window, y[i]))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ── Función de entrenamiento ─────────────────────────────────
def entrenar(model, nombre, train_dl, val_dl):
    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    best_val  = float("inf")
    best_path = RESULTS_DIR / f"modelo_{nombre.lower()}_4.pth"
    train_losses, val_losses = [], []

    print(f"\nEntrenando {nombre} — {sum(p.numel() for p in model.parameters()):,} parámetros")

    pbar = tqdm(range(1, EPOCHS + 1), desc=nombre, unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for epoch in pbar:
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(xb), yb).item()
        v_loss /= len(val_dl)
        scheduler.step(v_loss)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            pbar.set_postfix(train=f"{t_loss:.5f}", val=f"{v_loss:.5f}", mejor="✓")
        else:
            pbar.set_postfix(train=f"{t_loss:.5f}", val=f"{v_loss:.5f}")

    pbar.close()
    print(f"{nombre} — mejor val loss: {best_val:.6f} | guardado: {best_path}")

    # Gráfica de curvas de loss
    plot_path = RESULTS_DIR / f"loss_{nombre.lower()}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train", linewidth=1.5)
    ax.plot(val_losses,   label="Val",   linewidth=1.5)
    ax.axhline(best_val, color="gray", linestyle="--", linewidth=1, label=f"Mejor val={best_val:.5f}")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Curva de entrenamiento — {nombre}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Gráfica guardada en {plot_path}")

    return best_path


# ── Modelos ──────────────────────────────────────────────────
class LSTMDrone(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_layers=2, output_size=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(64, output_size))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MambaDrone(nn.Module):
    def __init__(self, input_size=18, d_model=128, n_layers=2, output_size=4):
        super().__init__()
        from mamba_ssm import Mamba
        self.input_proj   = nn.Linear(input_size, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fc    = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(64, output_size))
    def forward(self, x):
        x = self.input_proj(x)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            x = norm(x + mamba(x))
        return self.fc(x[:, -1, :])


# ── Main ─────────────────────────────────────────────────────
def main():
    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} filas | {df['episodio'].nunique()} episodios")

    # 1. Split PRIMERO (sin data leakage)
    episodios = df["episodio"].unique()
    np.random.seed(42)
    np.random.shuffle(episodios)
    n_val    = int(len(episodios) * VAL_SPLIT)
    ep_val   = episodios[:n_val]
    ep_train = episodios[n_val:]

    df_train = df[df["episodio"].isin(ep_train)].copy()
    df_val   = df[df["episodio"].isin(ep_val)].copy()

    # 2. Stats solo con train
    stats = {}
    for col in INPUT_COLS + OUTPUT_COLS:
        mean = df_train[col].mean()
        std  = df_train[col].std()
        stats[col] = [float(mean), float(std) if std > 1e-8 else 1.0]

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats guardadas en {STATS_PATH}")

    # 3. Normalizar con las mismas stats
    df_train_norm = df_train.copy()
    df_val_norm   = df_val.copy()
    for col in INPUT_COLS + OUTPUT_COLS:
        m, s = stats[col]
        df_train_norm[col] = (df_train[col] - m) / s
        df_val_norm[col]   = (df_val[col]   - m) / s

    train_ds = DroneDataset(df_train_norm, WINDOW_SIZE)
    val_ds   = DroneDataset(df_val_norm,   WINDOW_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(ep_train)} ep | {len(train_ds):,} muestras")
    print(f"Val:   {len(ep_val)} ep  | {len(val_ds):,} muestras")

    # Entrenar Mamba
    entrenar(MambaDrone(), "Mamba", train_dl, val_dl)

    print("\nListo. Modelo guardado en:", RESULTS_DIR)


if __name__ == "__main__":
    main()
