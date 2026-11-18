"""Dispatch the explicit, equivalence-verified representation-only backend."""
def run(spec,*args,**kw):
 if spec.get('compact_options',False):
  from .cache_compact import run as execute
 else:
  from .execution_guarded import run as execute
 return execute(spec,*args,**kw)
