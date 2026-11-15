"""Temporary idle-sleep prevention scoped to the running local campaign."""
import argparse,ctypes,json,os,time
from ctypes import wintypes
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',required=True,type=int);args=parser.parse_args()
    if os.name!='nt':raise SystemExit('Windows only')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.SetThreadExecutionState.argtypes=[wintypes.DWORD];kernel.SetThreadExecutionState.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x00100000,False,args.pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    out=Path(__file__).resolve().parents[2]/'runs/intervention_round6/power_guard.json'
    record=dict(campaign_pid=args.pid,guard_pid=os.getpid(),started_epoch=time.time(),active=False,scope='Prevent idle system sleep only while this campaign process exists;display may turn off;no persistent power-plan changes.')
    try:
        if not kernel.SetThreadExecutionState(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
        record['active']=True;out.write_text(json.dumps(record,indent=2),encoding='utf-8');print('Temporary system-awake request active for campaign',args.pid,flush=True)
        while True:
            result=kernel.WaitForSingleObject(handle,50000)
            if result==0:break
            if result!=258:raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.SetThreadExecutionState(0x80000000);kernel.CloseHandle(handle)
        record.update(active=False,finished_epoch=time.time());out.write_text(json.dumps(record,indent=2),encoding='utf-8')
if __name__=='__main__':main()
