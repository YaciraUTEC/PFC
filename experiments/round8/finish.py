"""Close only a genuinely accepted, reported, visually checked and portable study."""
import argparse,time
from .common import OUT,ROOT,read,save,sha,verify_round7,close
from .study import path,verify

def main(attempt=1):
 verify(attempt);folder=path(attempt);d=read(folder/'decision.json');assert d['accepted'];report=OUT/'report'/f'attempt_{attempt}';r=read(report/'resultados_finales.json');view=read(report/'presentation_verified.json')
 for p,v in r['plots'].items():assert sha(p)==v and view['plots'][p]==v
 names=['h0_reproduction','h2_base_options_v1_train','h2_base_options_v1_calibration','h6_factual_option_equivalence','h7_data_veto_factorial','h8_fast_execution_equivalence','h9_observed_support_validation','h8_checkpoint_selection']+[f'attempt_{attempt}_{x}' for x in ['prepared','evaluation','report','handoff']]
 names += ['h10_compact_first_decisions','h10_compact_physical','h10_compact_near_threshold','h10_compact_paired_timing','h10_compact_execution']
 for name in names:assert read(OUT/(name+'_conclusion.json'))['closed'],name
 assert read(folder/'tests.json')['exit_code']==0;bundle=read(OUT/f'attempt_{attempt}_handoff_conclusion.json');assert bundle['smoke_test']['passed'];assert sha(bundle['archive'])==bundle['archive_sha256'];integrity=verify_round7()
 sources=[p for p in folder.rglob('*.json') if p.is_file()]+[p for p in report.rglob('*') if p.is_file()];save(OUT/'FINAL_ARTIFACTS_MANIFEST.json',dict(files={str(p):sha(p) for p in sources},bundle_archive=bundle['archive'],bundle_archive_sha256=bundle['archive_sha256']))
 close('final_closure','Prospective gain and preservation evidence, strict landing tests, full-chain timing, complete regression, visual QA and relocated handoff all verified.',attempt=attempt,operational_gate=r['operational_gate'],round7_integrity=integrity)
 save(OUT/'ALL_COMPLETE.json',dict(complete=True,finished_epoch=time.time(),remaining_required_work=0,attempt=attempt,operational_gate=r['operational_gate'],decision=d,report=str(report/'CONCLUSION_GLOBAL_RONDA8.txt'),handoff=bundle['folder'],archive=bundle['archive']));save(OUT/'status.json',dict(stage='all_complete',status='complete',remaining_required_work=0));print('ROUND8_ALL_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);a=p.parse_args();main(a.attempt)
