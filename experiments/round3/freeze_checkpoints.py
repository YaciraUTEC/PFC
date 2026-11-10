from pathlib import Path
import json,hashlib,datetime,subprocess,platform,zipfile,io
import torch
import importlib.metadata as md
from .common import OUT,SEEDS,save

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy_sha(p):
    with zipfile.ZipFile(p) as z:state=torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=True)
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if k.startswith(('features_extractor.','mlp_extractor.policy_net.','action_net.','log_std')):h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    existing=OUT/'frozen_before_test.json'
    if existing.exists():
        for entry in json.loads(existing.read_text())['checkpoint_entries']:
            for name,expected in entry['weights_sha256'].items():assert sha(OUT/name)==expected,name
        print('Existing pre-test snapshot preserved;model hashes verified.',flush=True);return
    if any(OUT.rglob('*test.json')):raise RuntimeError('A pre-test snapshot cannot be created after flight-test results exist')
    root=OUT.parents[1];entries=[]
    for seed in SEEDS:
        sd=OUT/f'seed_{seed}';bc=policy_sha(sd/'bc.zip')
        for method in ['manual','csil','meirl','rime']:
            d=sd/method;complete=json.loads((d/'complete.json').read_text());assert complete['actual_steps']==262144
            selected=json.loads((d/'selection.json').read_text());weights={str(p.relative_to(OUT)):sha(p) for p in [sd/'bc.zip',d/'best.zip',d/'final.zip']+([] if method=='manual' else [d/'reward.pt'])}
            ph=policy_sha(d/'best.zip');entries.append(dict(seed=seed,method=method,selection=selected,weights_sha256=weights,selected_actor_sha256=ph,bc_actor_sha256=bc,selected_actor_equals_bc=ph==bc))
    paths=list((root/'experiments/round3').glob('*.py'))+[root/'experiments/models.py',root/'experiments/env.py',root/'experiments/local_env.ps1',root/'gym_pybullet_drones/control/DSLPIDControl.py',root/'gym_pybullet_drones/control/LegacyDSLPIDControl.py',root/'gym_pybullet_drones/control/RobustDSLPIDControl.py',root/'fallas/nominal_flight_env.py',root/'comparacion/comparar_base.py']
    save(OUT/'frozen_before_test.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_sha256={str(p.relative_to(root)):sha(p) for p in paths},checkpoint_entries=entries,system=platform.platform(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        packages={name:md.version(name) for name in ['torch','numpy','scipy','gymnasium','stable-baselines3','pybullet','matplotlib']},data_sha256=sha(OUT/'data.npz')))
    print('Frozen20 treatments before test; actor retained BC:',sum(r['selected_actor_equals_bc'] for r in entries),flush=True)
if __name__=='__main__':main()