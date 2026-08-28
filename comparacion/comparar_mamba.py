"""
Comparación PID vs Mamba en 5 trayectorias fijas.
Requiere: mamba-ssm (WSL/Linux con CUDA)
"""

import torch
import numpy as np
from pathlib import Path
from comparar_base import (
    cargar_stats, generar_waypoints, TRAYECTORIAS,
    nueva_env, redibujar, volar_pid, volar_modelo,
    imprimir_resumen, guardar, STATS_PATH,
    MambaDrone, COLOR_MAMBA,
)

_ROOT            = Path(__file__).parent.parent
MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
OUTPUT_FILE      = str(_ROOT / "results" / "comparacion_mamba.json")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Rojo = PID  |  Verde = Mamba")
    print("=" * 60)

    stats = cargar_stats(STATS_PATH)
    model = MambaDrone()
    model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device))
    model.to(device); model.eval()
    print(f"Mamba cargado: {MAMBA_MODEL_PATH}\n")

    # Un solo entorno para toda la sesión — evita segfault al reabrir GUI en WSL
    primer_A = np.array(TRAYECTORIAS[0]["A"])
    env = nueva_env(primer_A)

    resultados = []
    for i, tray in enumerate(TRAYECTORIAS):
        punto_A   = np.array(tray["A"])
        punto_B   = np.array(tray["B"])
        waypoints = generar_waypoints(punto_A, punto_B)

        env.INIT_XYZS = punto_A.reshape(1, 3)

        print(f"── Trayectoria {i+1}/{len(TRAYECTORIAS)}  "
              f"A={np.round(punto_A[:2],1)} → B={np.round(punto_B[:2],1)}")

        res_pid = volar_pid(env, punto_B, waypoints)
        print(f"  PID   (rojo):  {'✓' if res_pid['llego'] else '✗'} | "
              f"err={res_pid['error_final']:.3f} m | "
              f"long={res_pid['longitud']:.2f} m | "
              f"min_wp={res_pid['media_min_dist']:.3f} m | "
              f"pasos={res_pid['pasos']}")
        input("  Enter para Mamba...")

        redibujar(res_pid["posiciones"], (1.0, 0.3, 0.2), env.CLIENT)
        res_mamba = volar_modelo(env, punto_B, waypoints, model, stats, device, COLOR_MAMBA)
        print(f"  Mamba (verde): {'✓' if res_mamba['llego'] else '✗'} | "
              f"err={res_mamba['error_final']:.3f} m | "
              f"long={res_mamba['longitud']:.2f} m | "
              f"min_wp={res_mamba['media_min_dist']:.3f} m | "
              f"pasos={res_mamba['pasos']}")

        resultados.append({"trayectoria": i+1,
                            "punto_A": punto_A.tolist(), "punto_B": punto_B.tolist(),
                            "waypoints": [w.tolist() for w in waypoints],
                            "pid": res_pid, "mamba": res_mamba})
        input("  Enter para siguiente trayectoria...")

    imprimir_resumen(resultados, "Mamba", "mamba")
    guardar(resultados, OUTPUT_FILE)
    env.close()


if __name__ == "__main__":
    main()
