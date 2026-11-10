"""Activate existing C++ build tools; never install system-wide tools or caches."""
from pathlib import Path
import os,subprocess
_READY=False
def prepare_compiler():
    global _READY
    if _READY:return
    root=Path(__file__).resolve().parents[2]
    candidates=[Path(os.environ.get("ProgramFiles(x86)",r"C:\Program Files (x86)"))/"Microsoft Visual Studio/Installer/vswhere.exe",
                Path(os.environ.get("ProgramFiles",r"C:\Program Files"))/"Microsoft Visual Studio/Installer/vswhere.exe"]
    locator=next((p for p in candidates if p.is_file()),None)
    if locator is None:raise RuntimeError("Existing Visual Studio build tools could not be located")
    install=subprocess.check_output([str(locator),"-latest","-products","*","-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64","-property","installationPath"],text=True).strip()
    vcvars=Path(install)/"VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars.is_file():raise FileNotFoundError(vcvars)
    wrapper=root/"revision/compiler_env.cmd"
    wrapper.write_text('@echo off\ncall "'+str(vcvars)+'" >nul\nif errorlevel 1 exit /b 1\nset\n',encoding="ascii")
    raw=subprocess.check_output(["cmd.exe","/d","/c",str(wrapper)])
    for line in raw.decode("mbcs",errors="replace").splitlines():
        key,sep,value=line.partition("=")
        if sep and key:os.environ[key]=value
    dirs={"TEMP":"_tmp","TMP":"_tmp","TRITON_HOME":"_cache/triton",
      "TRITON_CACHE_DIR":"_cache/triton/cache","TORCHINDUCTOR_CACHE_DIR":"_cache/torchinductor",
      "TORCH_EXTENSIONS_DIR":"_cache/torch_extensions","CUDA_CACHE_PATH":"_cache/cuda"}
    for key,suffix in dirs.items():os.environ[key]=str(root/suffix)
    _READY=True
