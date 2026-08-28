"""
Entrena los modelos de detección de falla (LSTM vs Mamba) sobre
results/datos_falla_deteccion.csv (generado por generar_datos_falla.py).

Mismo patrón de ventana/entrenamiento que Modelos_deep/entrenar.py (DroneDataset
de 50 pasos con zero-padding, Adam + weight_decay + ReduceLROnPlateau + grad
clipping), pero con cabeza de clasificación+regresión en vez de regresión de RPM:
    falla_prob     (sigmoid, BCEWithLogitsLoss)
    fault_pct_pred (regresión, MSELoss, solo cuando falla_activa=1 en la etiqueta)

Uso (venv_mamba):
    cd /mnt/d/TesisI/gym-pybullet-drones
    /mnt/d/venv_mamba/bin/python3 fallas/entrenar_deteccion.py

Salida: results/modelo_deteccion_lstm.pth, results/modelo_deteccion_mamba.pth
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import INPUT_COLS  # noqa: E402

RESULTS_DIR = _ROOT / "results"
CSV_PATH    = RESULTS_DIR / "datos_falla_deteccion.csv"

WINDOW_SIZE = 50
BATCH_SIZE  = 128
VAL_SPLIT   = 0.2
EPOCHS      = 100
LR          = 1e-3
UMBRAL_PROB = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────
class FaultDataset(Dataset):
    def __init__(self, df, window_size):
        self.samples = []
        for ep in df["episodio"].unique():
            ep_df = df[df["episodio"] == ep].reset_index(drop=True)
            X  = ep_df[INPUT_COLS].values.astype(np.float32)
            ya = ep_df["falla_activa"].values.astype(np.float32)
            yp = ep_df["fault_pct"].values.astype(np.float32)
            for i in range(len(X)):
                if i >= window_size:
                    window = X[i - window_size:i]
                else:
                    pad    = np.zeros((window_size - i, X.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, X[:i]]) if i > 0 else np.zeros((window_size, X.shape[1]), dtype=np.float32)
                self.samples.append((window, ya[i], yp[i]))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        x, ya, yp = self.samples[idx]
        return torch.tensor(x), torch.tensor(ya), torch.tensor(yp)


# ── Modelos: mismo tronco que LSTMDrone/MambaDrone, cabeza de detección ──────
class LSTMDetector(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.trunk      = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.1))
        self.falla_head = nn.Linear(64, 1)
        self.pct_head   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.trunk(out[:, -1, :])
        return self.falla_head(h).squeeze(-1), self.pct_head(h).squeeze(-1)


class MambaDetector(nn.Module):
    def __init__(self, input_size=18, d_model=128, n_layers=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.input_proj   = nn.Linear(input_size, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.norms       = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.trunk       = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1))
        self.falla_head  = nn.Linear(64, 1)
        self.pct_head    = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            x = norm(x + mamba(x))
        h = self.trunk(x[:, -1, :])
        return self.falla_head(h).squeeze(-1), self.pct_head(h).squeeze(-1)


# ── Entrenamiento ─────────────────────────────────────────────
def entrenar(model, nombre, train_dl, val_dl):
    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    bce       = nn.BCEWithLogitsLoss()
    best_val  = float("inf")
    best_path = RESULTS_DIR / f"modelo_deteccion_{nombre.lower()}.pth"

    print(f"\nEntrenando detector {nombre} — "
          f"{sum(p.numel() for p in model.parameters()):,} parámetros")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for xb, ya, yp in train_dl:
            xb, ya, yp = xb.to(DEVICE), ya.to(DEVICE), yp.to(DEVICE)
            optimizer.zero_grad()
            logit_activa, pred_pct = model(xb)
            loss = bce(logit_activa, ya)
            mask = ya > 0.5
            if mask.any():
                loss = loss + nn.functional.mse_loss(pred_pct[mask], yp[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

        model.eval()
        v_loss, correctos, total, err_pct_sum, err_pct_n = 0.0, 0, 0, 0.0, 0
        with torch.no_grad():
            for xb, ya, yp in val_dl:
                xb, ya, yp = xb.to(DEVICE), ya.to(DEVICE), yp.to(DEVICE)
                logit_activa, pred_pct = model(xb)
                loss = bce(logit_activa, ya)
                mask = ya > 0.5
                if mask.any():
                    loss = loss + nn.functional.mse_loss(pred_pct[mask], yp[mask])
                    err_pct_sum += torch.abs(pred_pct[mask] - yp[mask]).sum().item()
                    err_pct_n   += mask.sum().item()
                v_loss += loss.item()
                pred_bin   = (torch.sigmoid(logit_activa) > UMBRAL_PROB).float()
                correctos += (pred_bin == ya).sum().item()
                total     += ya.numel()
        v_loss /= len(val_dl)
        scheduler.step(v_loss)
        acc = correctos / total
        mae = err_pct_sum / max(err_pct_n, 1)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            marca = "✓"
        else:
            marca = " "

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}/{EPOCHS} train={t_loss:.4f} val={v_loss:.4f} "
                  f"acc={acc:.3f} mae_pct={mae*100:.2f}% {marca}")

    print(f"{nombre} — mejor val loss: {best_val:.4f} | guardado: {best_path}")
    return best_path


def evaluar_latencia(model, df, ep_val):
    """Para cada episodio de validación: pasos desde el inicio real de la falla
    hasta la primera predicción positiva correcta (ventana deslizante de 50)."""
    model.eval()
    latencias = []
    for ep in ep_val:
        ep_df = df[df["episodio"] == ep].reset_index(drop=True)
        X  = ep_df[INPUT_COLS].values.astype(np.float32)
        ya = ep_df["falla_activa"].values
        if ya.sum() == 0:
            continue
        t_onset = int(np.argmax(ya))  # primer paso con falla_activa=1

        ventana = np.zeros((WINDOW_SIZE, X.shape[1]), dtype=np.float32)
        t_deteccion = None
        with torch.no_grad():
            for t in range(t_onset, len(X)):
                ventana = np.vstack([ventana[1:], X[t:t+1]])
                x = torch.tensor(ventana).unsqueeze(0).to(DEVICE)
                logit_activa, _ = model(x)
                if torch.sigmoid(logit_activa).item() > UMBRAL_PROB:
                    t_deteccion = t
                    break
        if t_deteccion is not None:
            latencias.append(t_deteccion - t_onset)
    return latencias


def main():
    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} filas | {df['episodio'].nunique()} episodios")

    episodios = df["episodio"].unique()
    np.random.seed(42)
    np.random.shuffle(episodios)
    n_val    = int(len(episodios) * VAL_SPLIT)
    ep_val   = episodios[:n_val]
    ep_train = episodios[n_val:]

    df_train = df[df["episodio"].isin(ep_train)].copy()
    df_val   = df[df["episodio"].isin(ep_val)].copy()

    # Normalizar INPUT_COLS con stats calculadas solo sobre train (sin leakage)
    stats = {}
    for col in INPUT_COLS:
        m, s = df_train[col].mean(), df_train[col].std()
        stats[col] = (float(m), float(s) if s > 1e-8 else 1.0)
    for col in INPUT_COLS:
        m, s = stats[col]
        df_train[col] = (df_train[col] - m) / s
        df_val[col]   = (df_val[col] - m) / s

    stats_path = RESULTS_DIR / "stats_deteccion.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats de normalización guardadas en {stats_path} "
          f"(necesarias para usar el detector en inferencia)")

    train_ds = FaultDataset(df_train, WINDOW_SIZE)
    val_ds   = FaultDataset(df_val,   WINDOW_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {len(ep_train)} ep | {len(train_ds):,} muestras")
    print(f"Val:   {len(ep_val)} ep  | {len(val_ds):,} muestras")

    resultados = {}
    for nombre, modelo in [("LSTM", LSTMDetector()), ("Mamba", MambaDetector())]:
        entrenar(modelo, nombre, train_dl, val_dl)
        latencias = evaluar_latencia(modelo, df_val, ep_val)
        resultados[nombre] = latencias
        if latencias:
            print(f"{nombre} — latencia de detección: "
                  f"media={np.mean(latencias):.1f} pasos "
                  f"({np.mean(latencias)/48*1000:.0f} ms) | mediana={np.median(latencias):.1f} pasos")

    print("\nListo. Modelos guardados en:", RESULTS_DIR)


if __name__ == "__main__":
    main()
