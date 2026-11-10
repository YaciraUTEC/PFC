"""Shared linear reward contract for trained and evaluated controllers."""
import json
import numpy as np

def load_linear_weights(path):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    weights = np.asarray(data["weights"], dtype=np.float64)
    if data.get("weights_space") == "scaled_features":
        scale = np.asarray(data["feature_scale"], dtype=np.float64)
        if scale.shape != weights.shape or np.any(scale <= 0):
            raise ValueError("Missing or invalid feature_scale in reward artifact")
        weights = weights / scale
    if np.any(weights < 0):
        raise ValueError("Unsafe legacy reward signs; learn a new constrained reward with its scales")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Nonfinite reward weights")
    return weights, data["feature_names"]

def exposed(info):
    return info.get("t_falla_seg") is not None and info.get("pasos_post_falla", 1) > 0
