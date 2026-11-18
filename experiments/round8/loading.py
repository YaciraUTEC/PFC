"""Versioned loader keeps all previous version2 policies unchanged."""
from .policy import PreservePolicy
class Policy:
 @classmethod
 def from_spec(cls,spec):
  if spec.get('fast_execution'):
   from .fast_policy import FastFactory
   return FastFactory.from_spec(spec)
  if spec.get('version')==3:
   from .policy_v3 import EntryPreserveOptions
   return EntryPreserveOptions.from_spec(spec)
  return PreservePolicy.from_spec(spec)
