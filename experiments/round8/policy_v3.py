"""Support-aligned entry veto; committed raw option has no internal Q cancellation."""
from .policy_v2 import PreserveOptions

class EntryPreserveOptions(PreserveOptions):
 def build_options(self,obs,context,raw,on):
  self._entry_native=on
  return super().build_options(obs,context,raw,on)
 def assess(self,x,delta,margin=0.):
  allowed,metrics=super().assess(x,delta,margin)
  # Labels for raw24half are collected only where the native proposal is ON at
  # entry. This condition restricts entry, not the next24 feedback actions.
  return allowed & self._entry_native[:,None],metrics
