"""Dispatch without altering the already evaluated version2 inference sources."""
def run(spec,*args,**kw):
 if spec.get('fast_execution'):
  from .cache_fast import run as execute
 elif spec.get('version')==3:
  from .cache_v3 import run as execute
 else:
  from .cache import run as execute
 return execute(spec,*args,**kw)
