"""Frozen evaluation-only implementation settings, selected by equivalence checks."""
from .common import OUT,read
DEFAULT=dict(validated=False,inference_device='cpu',certification_device='cpu',fast_gtrxl=False,workers=8,certification_batch=8)
def settings():
    path=OUT/'execution_settings.json'
    return dict(DEFAULT,**read(path)) if path.exists() else dict(DEFAULT)
