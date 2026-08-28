"""
Reevalúa Mamba+PPO (results/ppo_mamba_5, el mismo modelo ya comparado en
falla_tesis.py con instante de falla fijo) con el instante de falla
ALEATORIO en cada corrida, para mostrar qué tan bien generaliza fuera del
instante fijo (T_FALLA_SEG=3.0s) con el que se entrenó/comparó originalmente.

No modifica ni reemplaza falla_tesis.json — guarda un archivo aparte.

Uso (venv_mamba):
    cd /mnt/d/TesisI/gym-pybullet-drones
    /mnt/d/venv_mamba/bin/python3 fallas/falla_tesis_robustez.py
    /mnt/d/venv_mamba/bin/python3 fallas/falla_tesis_robustez.py --repeticiones 3

Salida: results/falla_tesis_robustez.json
"""
import argparse
import sys, json
import numpy as np
import torch
from pathlib import Path
from collections import deque

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "comparacion"))

from stable_baselines3 import PPO  # noqa: E402
from comparar_base import (  # noqa: E402
    cargar_stats, generar_waypoints, normalizar_estado, desnormalizar_accion,
    nueva_env, MambaDrone,
    STATS_PATH, INPUT_COLS, WINDOW_SIZE, HOVER_RPM, CTRL_FREQ,
    UMBRAL_WAYPOINT, DURACION_FALLA_SEG, MIN_RPM, MAX_RPM, MOTOR_FALLA,
    T_FALLA_MIN, T_FALLA_MAX,
    es_caida, es_aterrizaje, TRAYECTORIAS,
)

MAMBA_MODEL_PATH = str(_ROOT / "results" / "modelo_mamba.pth")
PPO_PATH         = str(_ROOT / "results" / "ppo_mamba_5" / "best_model.zip")
OUTPUT_FILE      = str(_ROOT / "results" / "falla_tesis_robustez.json")
DELTA_MAX        = 1500

SEVERIDADES = {"5pct": 0.95, "10pct": 0.90, "15pct": 0.85}
TIPOS = ["Diagonal larga", "Recto norte", "Recto este", "Diagonal inversa", "Diagonal sur"]


def volar_ppo_robustez(env, waypoints, punto_B, mamba_model, ppo, stats, device,
                       severidad, t_falla_seg):
    t_falla_paso = int(t_falla_seg * CTRL_FREQ)
    obs, _ = env.reset()
    ventana = deque([np.zeros(len(INPUT_COLS), dtype=np.float32)] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
    action = np.ones((1, 4)) * HOVER_RPM
    pos_hist = []; wp_idx = 0; llego = False; aterrizo = False; cayo = False; t_caida = 0

    for paso in range(int(DURACION_FALLA_SEG * CTRL_FREQ)):
        obs, _, term, trunc, _ = env.step(action)
        pos = obs[0][0:3].copy()
        pos_hist.append(pos.tolist())
        ta = waypoints[min(wp_idx, len(waypoints) - 1)]
        ventana.append(normalizar_estado(obs, ta, stats))
        x = torch.tensor(np.array(ventana), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = mamba_model(x).cpu().numpy()[0]
        base_rpm = np.clip(desnormalizar_accion(pred, stats), MIN_RPM, MAX_RPM)

        if paso >= t_falla_paso:
            fault_pct  = 1.0 - severidad
            fault_info = np.array([1.0, fault_pct], dtype=np.float32)
            obs_ppo    = np.concatenate([ventana[-1], pred, fault_info])
            delta, _   = ppo.predict(obs_ppo, deterministic=True)
            rpm        = np.clip(base_rpm + delta * DELTA_MAX, MIN_RPM, MAX_RPM)
            rpm[MOTOR_FALLA] *= severidad
            action = np.clip(rpm, MIN_RPM, MAX_RPM).reshape(1, 4)
        else:
            action = base_rpm.reshape(1, 4)

        if es_caida(obs, pos, paso):      cayo = True;     t_caida = paso; break
        if es_aterrizaje(obs, pos, paso): aterrizo = True; break
        if np.linalg.norm(ta - pos) < UMBRAL_WAYPOINT:
            wp_idx += 1
            if wp_idx >= len(waypoints): llego = True; break
        if term or trunc: break

    return {
        "llego": llego, "aterrizo": aterrizo, "cayo": cayo,
        "resultado": "llego" if llego else ("aterrizo" if aterrizo else "cayo"),
        "t_caida_seg": round(t_caida / CTRL_FREQ, 2) if cayo else None,
        "t_falla_seg": round(t_falla_seg, 2),
        "error_final": float(np.linalg.norm(np.array(pos_hist[-1]) - punto_B)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeticiones", type=int, default=3,
                        help="corridas por trayectoria x severidad, con instante de falla "
                             "distinto cada vez (para promediar sobre la aleatoriedad)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRobustez Mamba+PPO ante instante de falla aleatorio "
          f"[{T_FALLA_MIN}s, {T_FALLA_MAX}s] | device={device}")

    stats = cargar_stats(STATS_PATH)
    mamba_model = MambaDrone()
    mamba_model.load_state_dict(torch.load(MAMBA_MODEL_PATH, map_location=device, weights_only=True))
    mamba_model.to(device); mamba_model.eval()
    ppo = PPO.load(PPO_PATH)

    resultados = []
    for i, (tray, tipo) in enumerate(zip(TRAYECTORIAS, TIPOS)):
        punto_A = np.array(tray["A"]); punto_B = np.array(tray["B"])
        waypoints = generar_waypoints(punto_A, punto_B)
        print(f"\n── T{i+1}: {tipo}")

        for nombre, sev in SEVERIDADES.items():
            pct = round((1 - sev) * 100)
            for rep in range(args.repeticiones):
                t_falla_seg = float(np.random.uniform(T_FALLA_MIN, T_FALLA_MAX))
                env = nueva_env(punto_A)
                res = volar_ppo_robustez(env, waypoints, punto_B, mamba_model, ppo, stats,
                                         device, sev, t_falla_seg)
                env.close()
                print(f"  [{pct}% @ {t_falla_seg:.1f}s rep{rep}] "
                      f"{res['resultado']:8s} err={res['error_final']:.3f}m")
                resultados.append({
                    "trayectoria": i + 1, "tipo": tipo, "perdida_pct": pct,
                    "repeticion": rep, **res,
                })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(resultados, f, indent=2)

    tasa_exito = np.mean([r["resultado"] in ("llego", "aterrizo") for r in resultados])
    print(f"\nTasa de éxito (llegó o aterrizó) con instante aleatorio: {tasa_exito*100:.1f}%")
    print(f"Guardado en {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
