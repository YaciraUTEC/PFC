"""Final data-integrity audit and measured campaign times."""
from pathlib import Path
import json,hashlib,datetime,zipfile,subprocess,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/irl_round3';TASK=ROOT.parent
SEEDS=[11,22,33,44,55];METHODS=['manual','csil','meirl','rime'];LEVELS=[0.,.02,.05,.1,.15,.2,.25]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(rows,seeds):
    expected={(l,s) for l in LEVELS for s in seeds};keys={(r['level'],r['seed']) for r in rows}
    assert keys==expected and len(rows)==len(expected)
    for r in rows:
        assert r['outcome'] in ['mission','crash','landing','timeout']
        assert bool(r['mission_success'])==(r['outcome']=='mission')
        assert 0<r['episode_steps']<=960
        assert np.isfinite([r['tracking_rmse'],r['effort'],r['max_tilt'],r['saturation']]).all()
        if r['level']>0 and r['mission_success']:assert r['exposed'] and r['pasos_post_falla']>0
        if r['level']==0:assert not r['exposed']
    return len(rows)
def main():
    assert read(OUT/'status.json')['stage']=='complete';frozen=read(OUT/'frozen_before_test.json');checks={};total=0
    for x in frozen['checkpoint_entries']:
        assert read(OUT/f"seed_{x['seed']}"/x['method']/'selection.json')==x['selection']
        assert read(OUT/f"seed_{x['seed']}"/x['method']/'complete.json')['actual_steps']==262144
        for name,expected in x['weights_sha256'].items():
            if name not in checks:checks[name]=sha(OUT/name);assert checks[name]==expected,name
    b=read(OUT/'baselines_test.json')
    for key in set((r['controller'],r['condition']) for r in b):total+=validate([r for r in b if (r['controller'],r['condition'])==key],range(200000,200100))
    total+=validate(read(OUT/'baseline_exact_secondary.json'),range(200100,200120))
    for seed in SEEDS:
        for method in ['bc']+METHODS:
            d=OUT/f'seed_{seed}' if method=='bc' else OUT/f'seed_{seed}'/method
            total+=validate(read(d/('bc_test.json' if method=='bc' else 'test.json')),range(200000,200100))
            total+=validate(read(d/('bc_exact_test.json' if method=='bc' else 'exact_test.json')),range(200100,200120))
            if method!='bc':total+=validate(read(d/'final_test.json'),range(200000,200030))
            for stem in ['test','exact_test']:
                p=d/(stem+'_timing.json')
                if p.exists() and 'reused_from' in read(p):assert read(d/(stem+'.json'))==read(OUT/read(p)['reused_from'])
    assert total==30240
    xml=ET.parse(TASK/'revision/pruebas_round3.xml').getroot();tests=sum(int(x.attrib.get('tests',0)) for x in xml.iter('testsuite'));failures=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in xml.iter('testsuite'));assert failures==0
    assert len(read(OUT/'reward_validation_diagnostics.json')['diagnostics'])==15
    assert len(read(OUT/'policy_validation_diagnostics.json')['rows'])==25
    # Measured elapsed campaign time; parallel durations are not added as wall time.
    starts=[datetime.datetime.fromisoformat(read(p)['start_utc']).timestamp() for p in OUT.glob('execution*.json')]
    completions=[OUT/f'seed_{s}'/m/'complete.json' for s in SEEDS for m in METHODS]
    train_end=max(p.stat().st_mtime for p in completions);train_start=min(starts)
    finish=datetime.datetime.fromisoformat(read(OUT/'status.json')['finished_utc']).timestamp()
    eval_start=(TASK/'revision/evaluacion_final_round3.log').stat().st_ctime
    data=read(OUT/'data_episodes.json');pilot=read(TASK/'runs/controller_round3_pilot.json')
    timing=dict(controller_pilot_seconds=pilot['seconds'],demonstrations_seconds=data['seconds'],
        shared_bc_seconds=sum(read(OUT/f'seed_{s}/bc_timing.json')['seconds'] for s in SEEDS),
        training_campaign_elapsed_seconds=train_end-train_start,evaluation_elapsed_seconds=finish-eval_start,
        training_start_utc=datetime.datetime.fromtimestamp(train_start,datetime.timezone.utc).isoformat(),training_end_utc=datetime.datetime.fromtimestamp(train_end,datetime.timezone.utc).isoformat(),
        evaluation_start_utc=datetime.datetime.fromtimestamp(eval_start,datetime.timezone.utc).isoformat(),evaluation_end_utc=datetime.datetime.fromtimestamp(finish,datetime.timezone.utc).isoformat(),
        interpretation='Wall times include validation,parallelism,restarts and infrastructure work;paper review and implementation were not separately timed. Evaluation clock includes the intentional restart to reuse identical actors.')
    save(OUT/'campaign_timing.json',timing)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,tests=tests,model_case_records=total,model_files_unchanged_after_test=len(checks),weights_sha256=checks,selection_unchanged=True,scenario_ids_unique_and_complete=True,all_nonzero_successes_exposed=True,counts=read(OUT/'evaluation_counts.json'))
    save(OUT/'final_audit.json',result)
    # Current reviewable code overlay; excludes models,data,git metadata and caches.
    paths=set(subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMR'],cwd=ROOT,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines())
    active=[]
    for name in sorted(paths):
        p=ROOT/name
        if p.is_file() and p.suffix.lower() in ['.py','.ps1','.txt','.md','.toml','.yml','.yaml'] and not set(p.relative_to(ROOT).parts)&{'runs','results','__pycache__','.git','data','datasets'}:active.append(p)
    archive=TASK/'CODIGO_DSLPID_IRL_PRIMERA_PRUEBA.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in active:z.write(p,str(p.relative_to(ROOT)))
        for name in ['protocol.json','frozen_before_test.json','final_audit.json','analysis.json','campaign_timing.json','evaluation_counts.json']:z.write(OUT/name,'revision_metadata/'+name)
    save(OUT/'release_manifest.json',dict(code_archive=str(archive),code_archive_sha256=sha(archive),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in active},data_sha256=sha(OUT/'data.npz')))
    report=TASK/'INFORME_DSLPID_IRL_PRIMERA_PRUEBA.txt';text=report.read_text(encoding='utf-8');marker='\nVERIFICACIÓN FINAL Y TIEMPO DE CAMPAÑA\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+f"{tests} pruebas automatizadas aprobadas;30240 registros modelo-caso completos y únicos por su escenario;{len(checks)} archivos de modelos sin cambios después de congelarlos. Todas las misiones exitosas con falla realmente quedaron expuestas. Las selecciones por validación permanecieron sin cambios.\n"
    text+=f"Campaña de entrenamiento: {timing['training_campaign_elapsed_seconds']/60:.2f} minutos de reloj. Evaluación final: {timing['evaluation_elapsed_seconds']/60:.2f} minutos de reloj,incluido el reinicio para eliminar simulaciones idénticas. Piloto:{pilot['seconds']/60:.2f}min;demostraciones:{data['seconds']/60:.2f}min;clonación compartida de las cinco semillas:{timing['shared_bc_seconds']/60:.2f}min. La revisión bibliográfica y la implementación no se cronometraron por separado. Los tiempos paralelos de cada tratamiento no deben sumarse para inferir duración de reloj.\n"
    text+='La prueba con efectividad exacta muestra que un residual aprendido puede perjudicar a un controlador que ya compensa bien. La prioridad práctica es estimar la efectividad y activar el residual por necesidad observada. Antes de otro afinado con PPO conviene ampliar la validación y ensayar retención de la política clonada (penalización KL o de clonación y calentamiento del crítico);son propuestas para una nueva prueba,no mejoras demostradas aquí.\n'
    import re
    text=re.sub(r',(?=[A-Za-zÁÉÍÓÚáéíóúÑñ])',', ',text)
    for a,b in [('de7','de 7'),('de1','de 1'),('de2','de 2'),('de20','de 20'),('de2026','de 2026'),('Las210','Las 210'),('las168','las 168'),('LSTM64','LSTM de 64 unidades'),('historial32','historial de 32 pasos'),('ensemble3','ensemble de 3'),('segmentos32','segmentos de 32 pasos'),('semillas11','semillas 11'),('semilla11','semilla 11'),('semilla22','semilla 22'),('semilla33','semilla 33'),('semilla44','semilla 44'),('semilla55','semilla 55'),('NeurIPS2025','NeurIPS 2025'),('ICLR2025','ICLR 2025'),('uniforme0,45','uniforme 0,45'),('entre1,4 y16','entre 1,4 y 16')]:text=text.replace(a,b)
    report.write_text(text,encoding='utf-8');(OUT/'report.txt').write_text(text,encoding='utf-8')
    print(json.dumps(dict(passed=True,tests=tests,counts=result['counts'],timing=timing,archive=str(archive)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()