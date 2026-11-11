from pathlib import Path
import json,datetime
root=Path('runs/controlled_landing_round4')
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,json.JSONDecodeError):return {}
selections=sorted(p.stem.replace('_selection','') for p in (root/'hpo').glob('*_selection.json'))
active=[]
for p in sorted((root/'models').glob('*/*/progress.json'),key=lambda p:p.stat().st_mtime,reverse=True):
    if not (p.parent/'complete.json').exists():
        x=read(p);active.append(dict(model=p.parent.parent.name,seed=p.parent.name,steps=x.get('steps'),seconds=round(x.get('seconds',0))))
hpo=[]
for p in (root/'hpo').glob('*/progress.json'):
    if not (p.parent/'complete.json').exists():hpo.append(dict(candidate=p.parent.name,**read(p)))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hpo_ready=selections,hpo_active=hpo,
training_complete=sum(1 for p in (root/'models').glob('*/*/complete.json')),active=active,
training_campaign=read(root/'training_campaign_progress.json').get('completed'),
evaluation_jobs=read(root/'evaluation_campaign_progress.json').get('completed',('mild_selection_'+str(read(root/'selection_safety_progress.json').get('completed')) if (root/'selection_safety_progress.json').exists() else None)),
evaluation_total=read(root/'evaluation_campaign_progress.json').get('total'),
test_complete=(root/'test_complete.json').exists(),report_complete=(root/'INFORME_RONDA4.txt').exists())))
