
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent

INPUT_COLS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "roll",  "pitch", "yaw",
    "ang_x", "ang_y", "ang_z",
    "target_x", "target_y", "target_z",
    "err_x", "err_y", "err_z",
]
OUTPUT_COLS = ["motor_0", "motor_1", "motor_2", "motor_3"]
VAL_SPLIT = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(_ROOT / "results" / "datos_combinado_1600.csv"))
    parser.add_argument("--salida", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    episodios = df["episodio"].unique()
    np.random.seed(42)
    np.random.shuffle(episodios)
    n_val = int(len(episodios) * VAL_SPLIT)
    ep_train = episodios[n_val:]
    df_train = df[df["episodio"].isin(ep_train)]

    stats = {}
    for col in INPUT_COLS + OUTPUT_COLS:
        mean = df_train[col].mean()
        std = df_train[col].std()
        stats[col] = [float(mean), float(std) if std > 1e-8 else 1.0]

    with open(args.salida, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Episodios: {len(episodios)} (entrenamiento {len(ep_train)})")
    print(f"Guardado en {args.salida}")


if __name__ == "__main__":
    main()
