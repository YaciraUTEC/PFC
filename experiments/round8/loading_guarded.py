from .loading import Policy as Previous
class Policy:
 @classmethod
 def from_spec(cls,spec):
  if spec.get('version')==4:
   from .policy_guarded import GuardedFactory
   return GuardedFactory.from_spec(spec)
  return Previous.from_spec(spec)
