"""Portable handoff within the authorized task tree, verified by a relocated demo."""
import argparse, shutil, sys, subprocess, time, zipfile, os
from pathlib import Path
from .common import ROOT, TASK, OUT, read, save, sha, close
from .study import path, verify

def copy(src, dst):
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    assert src.is_relative_to(TASK.resolve()) and dst.is_relative_to(TASK.resolve())
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert sha(src) == sha(dst), 'Different preexisting handoff file ' + str(dst)
        return
    tmp = dst.with_suffix(dst.suffix + '.partial')
    shutil.copy2(src, tmp)
    tmp.replace(dst)

def verify_delivered_results(destination,reported,operational,attempt):
    from .report import counts
    from ..round7.evaluation import compare
    raw=destination/'EVIDENCIA/evaluacion';checks=[]
    for row in reported['summary']:
        b=read(raw/row['phase']/'baseline.json');r=read(raw/row['phase']/f'gate_{operational}/results.json')
        if row['subset']=='estimated_critical':
            r=dict(episodes=[x for x in r['episodes'] if x['condition']=='estimated' and x['level'] in [.295,.296]])
        actual=counts(r,b)
        for k,v in actual.items():assert row[k]==v,('Copied raw results do not reproduce report',row['phase'],row['subset'],k)
        checks.append(dict(phase=row['phase'],subset=row['subset'],cases=actual['cases'],passed=True))
    for label,phase in reported['decision']['phases'].items():
        b=read(raw/label/'baseline.json')
        for entry in phase['entries']:
            r=read(raw/label/f'gate_{entry["seed"]}/results.json');assert compare(r,b)==entry['metrics']
    baseline=read(raw/'primary/baseline.json');candidate=read(raw/'primary'/f'gate_{operational}/results.json');previous=read(raw/'primary/previous_native/results.json');models={'DSLPID':baseline,'GTrXL anterior':previous,'GTrXL preservacion/rescate':candidate};end=840128+(attempt-1)*1000000
    curve_base=dict(episodes=[x for x in baseline['episodes'] if x['seed']<end]);curve_candidate=dict(episodes=[x for x in candidate['episodes'] if x['seed']<end]);matched=reported['matched_previous_native_curve']
    assert matched['cases']==len(curve_base['episodes']) and matched['baseline_successes']==sum(x['stable_landing'] for x in curve_base['episodes'])
    assert matched['previous_native']==compare(previous,curve_base) and matched['operational']==compare(curve_candidate,curve_base)
    for row in reported['curve_table']:
        rr=[x for x in models[row['controller']]['episodes'] if x['seed']<end and x['condition']==row['condition'] and x['noise']==row['noise'] and round(100*x['level'],1)==row['failure_percent']]
        success=sum(x['stable_landing'] for x in rr);assert len(rr)==row['routes']==128 and success==row['successes'] and 100*success/len(rr)==row['success_percent']
    legacy=destination/'EVIDENCIA/legacy_regression'/f'attempt_{attempt}';actual_legacy=compare(read(legacy/'candidate.json'),read(legacy/'baseline.json'));assert actual_legacy==reported['known_legacy_regression']['metrics'] and actual_legacy['losses']==0
    probe=destination/'EVIDENCIA/short_pulse_probe72';probe_base=read(probe/'baseline.json');assert compare(read(probe/'results.json'),probe_base)==reported['short_pulse_probe']['metrics'];assert compare(read(probe/'learned.json'),probe_base)==reported['short_pulse_probe']['learned_metrics']
    import numpy as np
    for example in read(destination/'EVIDENCIA/short_pulse_counterexamples.json')['cases']:
        original=probe/example['baseline_trace_relative_to_probe'];changed=probe/example['naive_trace_relative_to_probe'];assert sha(original)==example['baseline_trace_sha256'] and sha(changed)==example['naive_trace_sha256']
        with np.load(original) as z:baseline_observation=z['observation'].copy()
        with np.load(changed) as z:observation=z['observation'].copy();tick=int(np.flatnonzero(z['policy_option_started'])[0])
        assert np.array_equal(observation[:tick+1],baseline_observation[:tick+1]),'Counterexample prefix mismatch in relocated evidence'
    result=dict(known_legacy_cases=actual_legacy['cases'],known_legacy_losses=actual_legacy['losses'],passed=True,summary_checks=checks,curve_rows=len(reported['curve_table']),all_formal_replica_metrics_equal=True,scope='All aggregate counts, rates and curve points recomputed from the copied per-case outcome files; no original simulator histories needed for this audit.')
    save(destination/'EVIDENCIA/raw_outcome_recalculation.json',result);return result


def main(attempt=1):
    f = verify(attempt)
    folder = path(attempt)
    decision = read(folder / 'decision.json')
    review=read(folder/'prefinal_review.json')
    for name,digest in review['checks_sha256'].items():assert sha(folder/name)==digest,'Pre-final receipt changed: '+name
    assert decision['accepted'], 'Do not package an unaccepted controller as the recommendation'
    report = OUT / 'report' / f'attempt_{attempt}'
    viewed = read(report / 'presentation_verified.json')
    reported = read(report / 'resultados_finales.json')
    for name, d in reported['plots'].items():
        assert sha(name) == d and viewed['plots'][name] == d
    destination = TASK / 'ENTREGA_NICOLE_RONDA8' / f'intento_{attempt}'
    assert destination.resolve().is_relative_to(TASK.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    for rel in f['sources']:
        copy(ROOT / rel, destination / 'PFC_round8' / rel)
    for src in f['artifacts']:
        copy(src, destination / Path(src).relative_to(TASK))
    for pattern in ['*.json','*.txt']:
        for src in (ROOT/'experiments').rglob(pattern):
            copy(src,destination/'PFC_round8'/src.relative_to(ROOT))
    for rel in ['results/stats_normalizacion.json','results/round6/phase1/phase1_conclusion.json']:
        copy(ROOT/rel,destination/'PFC_round8'/rel)
    for rel in ['PFC/runs/intervention_round6/data/route_000.npz','vendor/mamba/tests/ops/triton/test_mamba3_siso.py','vendor/mamba/mamba_ssm/ops/selective_scan_interface.py','vendor/mamba/LICENSE']:
        copy(TASK/rel,destination/rel)
    for name in ['report.py', 'handoff.py']:
        copy(ROOT / 'experiments/round8' / name, destination / 'PFC_round8/experiments/round8' / name)
    for tag in ['base_options_v1','active_v1','active_merged_v1']:
        for src in (OUT / 'data' / tag).rglob('*'):
            if src.is_file():
                copy(src, destination / 'PFC_round8' / src.relative_to(ROOT))
    for src in (OUT / 'factor_gates').rglob('*'):
        if src.is_file() and (src.name.endswith('_complete.json') or src.suffix == '.pt'):
            copy(src, destination / 'PFC_round8' / src.relative_to(ROOT))
    for src in report.rglob('*'):
        if src.is_file():
            copy(src, destination / 'PFC_round8' / src.relative_to(ROOT))
    for name in ['frozen.json', 'selection.json', 'allocations.json', 'decision.json', 'latency.json', 'preflight.json', 'tests.json', 'tests.log']:
        copy(folder / name, destination / 'EVIDENCIA' / name)
    copy(folder/'prefinal_review.json',destination/'EVIDENCIA/prefinal_review.json')
    for label in ['primary','confirmation','shifted','assigned_initial']+(['assigned_extension'] if decision['extended'] else []):
        phase=folder/label
        for src in phase.rglob('*.json'):
            if 'changed_outcome_traces' not in src.parts and 'baseline_histories' not in src.parts:
                copy(src,destination/'EVIDENCIA/evaluacion'/label/src.relative_to(phase))
    for pattern in ['AMENDMENT*.json','sequential*.json']:
        for src in folder.glob(pattern):
            copy(src,destination/'EVIDENCIA'/src.name)
    for src in (OUT/'legacy_regression'/f'attempt_{attempt}').rglob('*'):
        if src.is_file():copy(src,destination/'EVIDENCIA/legacy_regression'/f'attempt_{attempt}'/src.relative_to(OUT/'legacy_regression'/f'attempt_{attempt}'))
    ablation=OUT/'development/short_pulse_without_learned_gate/probe72'
    for src in ablation.rglob('*'):
        if src.is_file():copy(src,destination/'EVIDENCIA/short_pulse_probe72'/src.relative_to(ablation))
    copy(OUT/'short_pulse_counterexamples.json',destination/'EVIDENCIA/short_pulse_counterexamples.json')
    for src in OUT.glob('PLAN*.txt'):
        copy(src,destination/'EVIDENCIA'/src.name)
    for src in OUT.glob('*conclusion.json'):
        copy(src, destination / 'EVIDENCIA' / src.name)
    for src in OUT.glob('*receipt.json'):
        copy(src, destination / 'EVIDENCIA' / src.name)
    for rel in ['development/PROSPECTIVE_SEVERITY_SWEEP.json','development/PROSPECTIVE_SEVERITY_SWEEP_original_rationale.json','development/observed_support_v1/PROSPECTIVE_ORIGINAL_DATA_CONTROL.json','development/observed_support_v1/prospective_specs.json','development/checkpoint_selection/prospective_checkpoint_registry.json','h6_early_loss_diagnostics.json','h9_raw_loss_diagnostics.json','h9_same_state_gate_predictions.json','h9_development_tradeoff_audit.json','h9_support_tradeoff_causes.json','h9_observed_support_comparison_summary.json','h8_rejected_expanded_raw505_losses.json','baseline_contract.json']:
        copy(OUT/rel,destination/'EVIDENCIA'/rel)
    for rel in ['execution_optimization/attempt_1_before_compaction/latency.json','execution_optimization/compact_first_decisions.npz','execution_optimization/compact_scan_inputs.json']:
        copy(OUT/rel,destination/'EVIDENCIA'/rel)
    copy(OUT / 'runtime_versions.json', destination / 'EVIDENCIA/runtime_versions.json')
    copy(OUT / 'data_support_audit.json', destination / 'EVIDENCIA/data_support_audit.json')
    for name in ['expanded_option_support_audit.json','training_data_manifest.json','h6_first_decision_scan_equivalence.json','hardware.json']:
        copy(OUT/name,destination/'EVIDENCIA'/name)
    for src in [ROOT / 'experiments/LICENSE_gtrxl.txt', ROOT / 'experiments/round6/vendor_flash/LICENSE', ROOT / 'LICENSE']:
        if src.exists():
            copy(src, destination / 'PFC_round8' / src.relative_to(ROOT))
    for name in ['data.npz', 'data_episodes.json']:
        copy(ROOT / 'runs/irl_round3' / name, destination / 'PFC_round8/runs/irl_round3' / name)
    raw_recalculation=verify_delivered_results(destination,reported,reported['operational_gate'],attempt)
    recommended = read(report / 'controlador_recomendado.json')
    spec = dict(recommended['spec'])
    for field in ['gate', 'proposer']:
        spec[field] = str(Path(spec[field]).relative_to(TASK)).replace('\\', '/')
    cfg = dict(recommended, spec=spec, bundle_root='..')
    save(destination / 'CONFIG/controlador.json', cfg)
    versions = read(OUT / 'runtime_versions.json')
    wanted = ['numpy', 'scipy', 'pandas', 'torch', 'gymnasium', 'stable-baselines3', 'pybullet', 'matplotlib', 'pillow', 'transforms3d', 'einops', 'ncps', 'pytest', 'setuptools', 'casadi', 'control', 'cloudpickle']
    canonical=lambda name:name.lower().replace('_','-').replace('.','-')
    lookup = {canonical(k): v for k, v in versions.items()}
    req = []
    for name in wanted:
        version = lookup[canonical(name)]
        req.append(name + '==' + version)
    (destination / 'requirements_entorno_verificado.txt').write_text('\n'.join(req) + '\n', encoding='utf-8')
    # The legacy regression template is a fixture, not the delivered controller.
    # Keep its original conclusion in EVIDENCIA and relocate only model paths
    # in the separate fixture copy so regression cannot silently use originals.
    fixture_source=OUT/'h5_conservative_validation_conclusion.json';fixture=read(fixture_source)
    def relocate_fixture(value):
        if isinstance(value,dict):return {k:relocate_fixture(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate_fixture(v) for v in value]
        if isinstance(value,str):
            candidate=Path(value)
            if candidate.is_absolute() and candidate.is_relative_to(TASK):return str(destination/candidate.relative_to(TASK))
        return value
    fixture=relocate_fixture(fixture);fixture['portable_regression_fixture_only']=True;fixture['original_conclusion_sha256']=sha(fixture_source)
    save(destination/'PFC_round8/runs/gtrxl_round8/h5_conservative_validation_conclusion.json',fixture)
    gate_path=Path(recommended['spec']['gate'])
    metadata=read(gate_path.with_name(gate_path.stem+'_complete.json'))
    training_folder='active_merged_v1' if gate_path.parent.name=='active_v1' else 'base_options_v1'
    train_command='python -B -m experiments.round8.fit --seed '+str(cfg['operational_gate'])+' --folder runs/gtrxl_round8/data/'+training_folder+' --tag reproduccion'+('' if metadata.get('route_bootstrap',True) else ' --full-support')
    selected=read(folder/'selection.json');development=Path(selected['development_folder']);dev_result=read(development/f'seed_{cfg["operational_gate"]}/results.json');case_key=lambda x:tuple(x[k] for k in ['seed','level','condition','noise','clock']);old={case_key(x):x for x in read(development/'baseline.json')['episodes']};rescues=[x for x in dev_result['episodes'] if x['condition']=='estimated' and x['stable_landing'] and not old[case_key(x)]['stable_landing'] and x.get('policy_summary',{}).get('actual_motor_intervention_steps',0)>0];assert rescues
    demo=min(rescues,key=lambda x:(x['noise']!=1.,x['clock'] is not None,x['level']!=.295,x['seed'],x['level']));demo_case={k:demo[k] for k in ['seed','level','condition','noise','clock']};save(destination/'CASO_DEMOSTRACION.json',dict(case=demo_case,choice='Known development rescue, selected descriptively; not a fresh performance estimate. Portability test must exercise a nonzero neural intervention.'))
    lines = ['ENTREGA A NICOLE - CIERRE GTrXL PRESERVACION/RESCATE', '', 'Este paquete incluye codigo, modelos, datos nuevos de compuerta, criterios, pruebas, resultados y versiones. Incluye tambien EVIDENCIA/evaluacion con los resultados brutos por caso para recomputar las comparaciones. No incluye el entorno Conda ni los miles de historiales de simulacion; siguen completos en la carpeta original. No se ha realizado un vuelo real.', '', '1. Leer PFC_round8/runs/gtrxl_round8/report/attempt_' + str(attempt) + '/CONCLUSION_GLOBAL_RONDA8.txt y tabla_por_falla.csv.', '2. El modelo operativo es la compuerta ' + str(cfg['operational_gate']) + ' sobre el actor GTrXL201 congelado. Las otras cuatro son replicas de aprendizaje de autorizacion, no actores reentrenados.', '3. Para un vuelo de demostracion: ejecutar DEMOSTRACION.ps1 desde PowerShell. Usa por defecto el entorno existente E:/UTEC/Tesis/TesisNicol/conda_env/python.exe; en otro PC pasar -PythonExe con la ruta de un entorno compatible alojado en esta carpeta. No instala nada; el resultado se guarda en SALIDAS_DEMO/demostracion.json y no sobrescribe la evidencia original. El caso es un rescate observado en desarrollo, guardado en CASO_DEMOSTRACION.json; no es una nueva estimacion estadistica.', '4. Para volver a entrenar la familia elegida, desde PFC_round8: '+train_command+'. Repetir con semillas501..505. El comando refleja los datos y bootstrap del checkpoint entregado. Las particiones, temperaturas y seleccion de epoca originales estan en *_complete.json; los modelos originales se incluyen con SHA256. Para reproducir la decision del artefacto entregado, usar sus pesos originales y CONFIG/controlador.json.', '5. El actor, su veto interno, el encoder, las escalas y la opcion de intervencion forman parte del controlador. Cambiarlos invalida las etiquetas anteriores; no basta recalibrar una cabeza de probabilidad.', '6. Las pruebas formales ya inspeccionadas no se reutilizan como nuevas. Si se modifica el controlador, emplear rutas nuevas y la siguiente asignacion de alfa de experiments/round8/sequential.py. Los datos de desarrollo y calibracion no se presentan como prueba final.', '7. Cero aterrizajes perdidos observados no garantiza cero riesgo universal. La cota y su poblacion estan en decision.json y en el informe; la banda de los graficos es rango entre cinco compuertas, no IC.', '', 'El criterio de contacto y estabilizacion permanece intacto. No se trabajo con Mamba en esta ronda. No hay ejecucion automatica en hardware.', '', 'MANIFEST_SHA256.json permite verificar cada archivo del paquete. smoke_test_receipt.json registra una ejecucion desde la copia trasladada y su igualdad con la ejecucion original.']
    (destination / 'LEEME_NICOLE.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    ps = "param([string]$PythonExe = 'E:\\UTEC\\Tesis\\TesisNicol\\conda_env\\python.exe')\n$bundleRoot = $PSScriptRoot\n$env:TEMP = Join-Path $bundleRoot '_tmp'\n$env:TMP = $env:TEMP\n$env:PYTHONDONTWRITEBYTECODE = '1'\n$env:PYTHONUTF8 = '1'\n$env:OMP_NUM_THREADS = '1'\n$env:MKL_NUM_THREADS = '1'\n$env:MPLCONFIGDIR = Join-Path $bundleRoot '_cache\\matplotlib'\n$env:TORCH_HOME = Join-Path $bundleRoot '_cache\\torch'\n$env:XDG_CACHE_HOME = Join-Path $bundleRoot '_cache'\nNew-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null\nSet-Location -LiteralPath (Join-Path $bundleRoot 'PFC_round8')\n& $PythonExe -B -m experiments.round8.runtime --config (Join-Path $bundleRoot 'CONFIG\\controlador.json') --seed810085 --loss29.6 --condition estimated --noise1 --output (Join-Path $bundleRoot 'demostracion.json')\nif ($LASTEXITCODE -ne 0) { throw 'La demostracion no termino correctamente' }\n"
    ps = ps.replace('--seed810085', '--seed '+str(demo['seed'])).replace('--loss29.6', '--loss '+str(100*demo['level'])).replace('--noise1', '--noise '+str(demo['noise']))
    if demo['clock'] is not None:ps=ps.replace('--output ', '--clock '+str(demo['clock'])+' --output ',1)
    ps=ps.replace("$env:XDG_CACHE_HOME = Join-Path $bundleRoot '_cache'\n","$env:XDG_CACHE_HOME = Join-Path $bundleRoot '_cache'\n$env:PYTHONPATH = Join-Path $bundleRoot 'PFC_round8'\n$env:HF_HOME = Join-Path $bundleRoot '_cache/huggingface'\n$env:CUDA_CACHE_PATH = Join-Path $bundleRoot '_cache/cuda'\n$env:TRITON_HOME = Join-Path $bundleRoot '_cache/triton'\n$env:TRITON_CACHE_DIR = Join-Path $bundleRoot '_cache/triton/cache'\n$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $bundleRoot '_cache/torchinductor'\n$env:TORCH_EXTENSIONS_DIR = Join-Path $bundleRoot '_cache/torch_extensions'\nforeach ($cacheDir in @($env:HF_HOME,$env:CUDA_CACHE_PATH,$env:TRITON_HOME,$env:TRITON_CACHE_DIR,$env:TORCHINDUCTOR_CACHE_DIR,$env:TORCH_EXTENSIONS_DIR)) { New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null }\n")
    ps=ps.replace("Set-Location -LiteralPath (Join-Path $bundleRoot 'PFC_round8')","$demoOutputDir = Join-Path $bundleRoot 'SALIDAS_DEMO'\nNew-Item -ItemType Directory -Force -Path $demoOutputDir | Out-Null\nSet-Location -LiteralPath (Join-Path $bundleRoot 'PFC_round8')").replace("(Join-Path $bundleRoot 'demostracion.json')","(Join-Path $demoOutputDir 'demostracion.json')")
    (destination / 'DEMOSTRACION.ps1').write_text(ps.replace(chr(92) * 2, chr(92)), encoding='utf-8')
    relocated_env=dict(os.environ)
    relocated_env.update(PYTHONPATH=str(destination/'PFC_round8'),TEMP=str(destination/'_tmp'),TMP=str(destination/'_tmp'),MPLCONFIGDIR=str(destination/'_cache/matplotlib'),TORCH_HOME=str(destination/'_cache/torch'),XDG_CACHE_HOME=str(destination/'_cache'))
    for name,rel in {'HF_HOME':'huggingface','TRITON_HOME':'triton','TRITON_CACHE_DIR':'triton/cache','TORCHINDUCTOR_CACHE_DIR':'torchinductor','TORCH_EXTENSIONS_DIR':'torch_extensions','CUDA_CACHE_PATH':'cuda','PIP_CACHE_DIR':'pip'}.items():
        relocated_env[name]=str(destination/'_cache'/rel);Path(relocated_env[name]).mkdir(parents=True,exist_ok=True)
    (destination/'_tmp').mkdir(exist_ok=True)
    before = time.time()
    import_code="""import sys,json,importlib,hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve();modules={}
for name in ['gym_pybullet_drones','gym_pybullet_drones.control.DSLPIDControl','experiments.round6.env','experiments.round8.loading_compact']:
 m=importlib.import_module(name);p=Path(m.__file__).resolve();assert p.is_relative_to(root),(name,str(p));modules[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[2]).write_text(json.dumps(dict(passed=True,modules=modules,source_root=str(root)),indent=2),encoding='utf-8')
"""
    with (destination/'source_import_audit.log').open('w',encoding='utf-8') as handle:
        imports=subprocess.run([sys.executable,'-B','-c',import_code,str(destination/'PFC_round8'),str(destination/'source_import_receipt.json')],cwd=destination/'PFC_round8',env=relocated_env,stdout=handle,stderr=subprocess.STDOUT)
    assert imports.returncode==0,'Relocated imports escaped the bundle; inspect source_import_audit.log'
    assert read(destination/'source_import_receipt.json')['passed']
    original = destination / 'original_demo.json'
    relocated = destination / 'relocated_demo.json'
    args = ['-B', '-m', 'experiments.round8.runtime', '--seed', str(demo['seed']), '--loss', str(100*demo['level']), '--condition', demo['condition'], '--noise', str(demo['noise'])]
    if demo['clock'] is not None:args.extend(['--clock',str(demo['clock'])])
    with (destination / 'smoke_test.log').open('w', encoding='utf-8') as handle:
        a = subprocess.run([sys.executable, *args, '--config', str(report / 'controlador_recomendado.json'), '--output', str(original)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        assert a.returncode == 0
        b = subprocess.run([sys.executable, *args, '--config', str(destination / 'CONFIG/controlador.json'), '--output', str(relocated)], cwd=destination / 'PFC_round8', env=relocated_env, stdout=handle, stderr=subprocess.STDOUT)
        assert b.returncode == 0
    aa = read(original)
    bb = read(relocated)
    assert Path(aa['source_root']).resolve()==ROOT.resolve()
    assert Path(bb['source_root']).resolve()==(destination/'PFC_round8').resolve()
    assert Path(bb['task_root']).resolve()==destination.resolve()
    assert not aa['baseline']['stable_landing'] and aa['candidate']['stable_landing'],'Portable smoke test must reproduce an actual development rescue'
    largest = 0.0
    import numpy as np
    for name in ['baseline', 'candidate']:
        assert aa[name]['outcome'] == bb[name]['outcome'] and aa[name]['episode_steps'] == bb[name]['episode_steps']
        assert aa[name]['first_contact']==bb[name]['first_contact'] and aa[name]['stable_seconds']==bb[name]['stable_seconds']
        error = float(np.max(np.abs(np.asarray(aa[name]['terminal_position']) - bb[name]['terminal_position'])))
        largest = max(largest, error)
        assert error == 0.0
    shell=shutil.which('pwsh') or shutil.which('powershell');assert shell,'PowerShell required to verify the delivered launcher'
    with (destination/'powershell_launcher.log').open('w',encoding='utf-8') as handle:
        launcher=subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(destination/'DEMOSTRACION.ps1'),'-PythonExe',sys.executable],cwd=destination,env=relocated_env,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert launcher.returncode==0,'Delivered DEMOSTRACION.ps1 failed; inspect powershell_launcher.log'
    launched=read(destination/'SALIDAS_DEMO/demostracion.json');save(destination/'EVIDENCIA/demostracion_launcher_verificada.json',launched);assert Path(launched['source_root']).resolve()==(destination/'PFC_round8').resolve()
    for name in ['baseline','candidate']:
        assert launched[name]['outcome']==bb[name]['outcome'] and launched[name]['episode_steps']==bb[name]['episode_steps'];assert launched[name]['first_contact']==bb[name]['first_contact'] and launched[name]['stable_seconds']==bb[name]['stable_seconds'];np.testing.assert_array_equal(launched[name]['terminal_position'],bb[name]['terminal_position'])
    with (destination/'regression_tests.log').open('w',encoding='utf-8') as handle:
        regression=subprocess.run([sys.executable,'-B','-m','pytest','experiments/tests','-q','-p','no:cacheprovider'],cwd=destination/'PFC_round8',env=relocated_env,stdout=handle,stderr=subprocess.STDOUT)
    assert regression.returncode==0,'Relocated regression failed; inspect '+str(destination/'regression_tests.log')
    save(destination / 'smoke_test_receipt.json', dict(passed=True,actual_development_rescue=True,demo_case=demo_case,relocated_regression_exit_code=regression.returncode,powershell_launcher_exit_code=launcher.returncode,powershell_launcher_matches_relocated_python=True,project_imports_within_bundle=read(destination/'source_import_receipt.json')['passed'], seconds=time.time() - before, original_and_relocated_exit_codes=[a.returncode, b.returncode], maximum_terminal_position_difference=largest,first_contact_and_stability_equal=True, relocated_source_root=bb['source_root'],scope='Actual baseline and guarded GTrXL demo from relocated source and relative model configuration; subprocess source path and caches constrained to bundle.'))
    manifest = {str(p.relative_to(destination)).replace('\\', '/'): sha(p) for p in destination.rglob('*') if p.is_file() and p.name != 'MANIFEST_SHA256.json' and (not any((x in p.parts for x in ['_tmp', '_cache', 'SALIDAS_DEMO'])))}
    save(destination / 'MANIFEST_SHA256.json', dict(files=manifest, count=len(manifest), bytes=sum(((destination / p).stat().st_size for p in manifest))))
    archive = destination.parent / (destination.name + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in list(manifest) + ['MANIFEST_SHA256.json']:
            z.write(destination / name, arcname=destination.name + '/' + name)
    import hashlib
    with zipfile.ZipFile(archive,'r') as z:
        assert z.testzip() is None,'ZIP CRC validation failed'
        assert len(z.namelist())==len(manifest)+1
        for name,digest in manifest.items():assert hashlib.sha256(z.read(destination.name+'/'+name)).hexdigest()==digest,'ZIP content mismatch: '+name
    verify(attempt)
    close('attempt_' + str(attempt) + '_handoff', 'Portable source, models, data and evidence bundle tested from its relocated path.', folder=str(destination), archive=str(archive), archive_sha256=sha(archive),archive_contents_verified=True, smoke_test=read(destination / 'smoke_test_receipt.json'),raw_outcome_recalculation=raw_recalculation)
    print('NICOLE_HANDOFF', destination, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', type=int, default=1)
    a = p.parse_args()
    main(a.attempt)
