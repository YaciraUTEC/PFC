"""Temporary process-scoped idle sleep prevention; no persistent power changes."""
import ctypes,os,contextlib
@contextlib.contextmanager
def awake():
    if os.name!='nt':yield;return
    k=ctypes.windll.kernel32
    if not k.SetThreadExecutionState(0x80000001):raise ctypes.WinError()
    try:yield
    finally:k.SetThreadExecutionState(0x80000000)
