"""
Evaluación headless con las 5 trayectorias definidas en la tesis.
Corre PID, LSTM y Mamba sin GUI y guarda resultados en comparacion_tesis.json.

Uso (WSL):
    /mnt/d/venv_mamba/bin/python3 comparacion/comparar_tesis.py
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from comparar_base import (
    cargar_stats, generar_waypoints, nueva_env,
    normalizar_estado, desnormalizar_accion,
    volar_pid, volar_modelo,
    LSTMDrone, MambaDrone,
    STATS_PATH, imprimir_resumen, guardar,
)

LSTM_MODEL_PATH  = str(_ROOT / "results" / "modelo_lstm.pth")
MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
OUTPUT_FILE      = str(_ROOT / "results" / "comparacion_tesis.json")

Z_SUELO = 0.1

# ── Trayectorias de la tesis ──────────────────────────────────
TRAYECTORIAS_TESIS = [
    {"tipo": "Diagonal larga",   "A": [-1.5, -1.5, Z_SUELO], "B": [ 1.5,  1.5, Z_SUELO]},
    {"tipo": "Recto norte",      "A": [ 0.0, -1.5, Z_SUELO], "B": [ 0.0,  1.5, Z_SUELO]},
    {"tipo": "Recto este",       "A": [-1.5,  0.0, Z_SUELO], "B": [ 1.5,  0.0, Z_SUELO]},
    {"tipo": "Diagonal inversa", "A": [ 1.0, -1.0, Z_SUELO], "B": [-1.0,  1.0, Z_SUELO]},
    {"tipo": "Diagonal sur",     "A": [-1.0,  1.0, Z_SUELO], "B": [ 1.0, -1.0, Z_SUELO]},
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluación tesis — 5 trayectorias | device={device}")
    print("=" * 60)

    stats = cargar_stats(STATS_PATH)

    lstm_model = LSTMDrone()
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device,
                                          weights_only=True))
    lstm_model.to(device); lstm_model.eval()

    mamba_model = MambaDrone()
    mamba_model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device,
                                           weights_only=True))
    mamba_model.to(device); mamba_model.eval()

    primer_A = np.array(TRAYECTORIAS_TESIS[0]["A"])
    env = nueva_env(primer_A, gui=False)

    resultados = []
    for i, tray in enumerate(TRAYECTORIAS_TESIS):
        punto_A   = np.array(tray["A"])
        punto_B   = np.array(tray["B"])
        waypoints = generar_waypoints(punto_A, punto_B)
        env.INIT_XYZS = punto_A.reshape(1, 3)

        print(f"\n── T{i+1}: {tray['tipo']}  "
              f"A={punto_A[:2]} → B={punto_B[:2]}")

        res_pid   = volar_pid(env, punto_B, waypoints)
        print(f"  PID   : {'✓' if res_pid['llego'] else '✗'}  "
              f"err={res_pid['error_final']:.3f}m")

        res_lstm  = volar_modelo(env, punto_B, waypoints, lstm_model,  stats, device)
        print(f"  LSTM  : {'✓' if res_lstm['llego'] else '✗'}  "
              f"err={res_lstm['error_final']:.3f}m")

        res_mamba = volar_modelo(env, punto_B, waypoints, mamba_model, stats, device)
        print(f"  Mamba : {'✓' if res_mamba['llego'] else '✗'}  "
              f"err={res_mamba['error_final']:.3f}m")

        resultados.append({
            "trayectoria": i + 1,
            "tipo":        tray["tipo"],
            "punto_A":     punto_A.tolist(),
            "punto_B":     punto_B.tolist(),
            "waypoints":   [w.tolist() for w in waypoints],
            "pid":         res_pid,
            "lstm":        res_lstm,
            "mamba":       res_mamba,
        })

    env.close()

    print("\n")
    imprimir_resumen(resultados, "LSTM",  "lstm")
    imprimir_resumen(resultados, "Mamba", "mamba")
    guardar(resultados, OUTPUT_FILE)
    print(f"\nGuardado en {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
