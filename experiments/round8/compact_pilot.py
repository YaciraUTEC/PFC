"""Development-only timing pilot for exact option-table compaction."""
import torch
from . import prepare
from .common import OUT,read,save,close
from .loading_compact import Policy
from ..round7.power import awake

def main():
 torch.set_num_threads(1);chosen=read(OUT/'final/attempt_1/selection.json');entry=next(x for x in chosen['entries'] if x['seed']==chosen['operational_gate']);entry=dict(entry,spec=dict(entry['spec'],compact_options=True));chosen=dict(chosen,entries=[entry]);prepare.PreservePolicy=Policy;prepare.path=lambda attempt:OUT/'execution_optimization/compact_pilot'
 with awake():result=prepare.latency(1,chosen)
 original=read(OUT/'final/attempt_1/latency.json')['results'][-1];new=result['results'][-1];assert original['flights']==new['flights'],'Compaction changed a pilot flight outcome/step count';close('h10_compact_latency_pilot','Only admissible option rows evaluated, identical pilot flight outcomes/steps; development timing only, full first-decision and physical equivalence checks still required.',before_guarded_p95_ms=original['guarded_p95_ms'],after_guarded_p95_ms=new['guarded_p95_ms'],meets_budget=new['guarded_p95_ms']<1000/48)
if __name__=='__main__':main()
