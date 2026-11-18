def run(spec,*args,**kw):
 if spec.get('version')==4:
  from .cache_guarded import run as execute
 else:
  from .execution import run as execute
 return execute(spec,*args,**kw)
