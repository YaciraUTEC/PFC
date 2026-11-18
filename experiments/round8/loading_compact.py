"""Evaluate only the explicitly admissible option rows; actor and gate weights stay fixed.
The returned option IDs remain global; internal choice indices are a smaller table.
No observation, threshold, amplitude or feedback duration is changed.
"""
from .loading_guarded import Policy as Previous
class Policy:
 @classmethod
 def from_spec(cls,spec):
  model=Previous.from_spec(spec)
  if spec.get('compact_options',False):
   assert model.allowed_options is not None
   keep=[i for i,option in enumerate(model.option_ids) if option in model.allowed_options]
   assert keep,'No admissible option'
   model.option_ids=[model.option_ids[i] for i in keep]
   model.options=[model.options[i] for i in keep]
  return model
