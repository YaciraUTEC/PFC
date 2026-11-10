"""Build a source-backed local report from the six completed paired tests."""
from pathlib import Path
import json,csv,hashlib,subprocess,datetime,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import ROOT,ARCHS,TEST_LEVELS,save

LABELS={'baseline':'DSLPIDControl','lstm':'LSTM','mamba':'Mamba','mamba2':'Mamba-2',
        'mamba3':'Mamba-3 (SISO)','gtrxl':'GTrXL','ltc':'LTC'}

def wilson(success,n,z=1.959963984540054):
    p=success/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-h),float(c+h)

def main():
    out=ROOT/'runs/dslpid_comparison_v2';dest=ROOT.parent/'resultados';dest.mkdir(exist_ok=True)
    data={'baseline':json.loads((out/'baseline_paired.json').read_text(encoding='utf-8'))}
    configs={};training={};reward_hash=hashlib.sha256((out/'reward.json').read_bytes()).hexdigest()
    for arch in ARCHS:
        data[arch]=json.loads((out/arch/'paired_test.json').read_text(encoding='utf-8'))
        training[arch]=json.loads((out/arch/'test.json').read_text(encoding='utf-8'))
        configs[arch]=json.loads((out/arch/'config.json').read_text(encoding='utf-8'))
        assert configs[arch]['reward_sha256']==reward_hash
        assert training[arch]['actual_steps']==524288,(arch,training[arch]['actual_steps'])
        assert configs[arch]['controller']=='DSLPIDControl'
        common=['history','steps','learning_rate_start','learning_rate_end','n_steps','n_envs','batch_size',
                'n_epochs','entropy_coef','gamma','gae_lambda','clip_range','target_kl','max_grad_norm','seed']
        for key in common:assert configs[arch][key]==configs['lstm'][key],(arch,key)
    for name,result in data.items():
        assert result['protocol']=='paired_seed_per_episode_v1'
        for level in TEST_LEVELS:
            rows=sorted([r for r in result['episodes'] if r['level']==level],key=lambda r:r['episode'])
            assert [r['scenario_seed'] for r in rows]==list(range(70000,70050))
        nominal=[r['mission_success'] for r in result['episodes'] if r['level']==0]
        assert nominal==[r['mission_success'] for r in data['baseline']['episodes'] if r['level']==0]
    table=[];paired=[]
    for name,result in data.items():
        for summary in result['summary']:
            n=summary['n'];success=round(summary['mission_success']*n);lo,hi=wilson(success,n)
            table.append(dict(model=LABELS[name],rpm_loss_pct=summary['level']*100,
              thrust_loss_pct=(1-(1-summary['level'])**2)*100,attempts=n,mission_success_count=success,
              mission_success_pct=100*summary['mission_success'],wilson95_low_pct=100*lo,wilson95_high_pct=100*hi,
              exposed=summary['exposed_n'],unexposed=summary['unexposed_n'],crash_pct=100*summary['crash']))
            if name!='baseline':
                base={r['scenario_seed']:r for r in data['baseline']['episodes'] if r['level']==summary['level']}
                rr=[r for r in result['episodes'] if r['level']==summary['level']]
                gains=sum(r['mission_success'] and not base[r['scenario_seed']]['mission_success'] for r in rr)
                losses=sum(not r['mission_success'] and base[r['scenario_seed']]['mission_success'] for r in rr)
                paired.append(dict(model=LABELS[name],rpm_loss_pct=summary['level']*100,
                    improved_routes=gains,worsened_routes=losses,net_improvement_points=100*(gains-losses)/n))
    for filename,rows in [('comparacion.csv',table),('diferencias_pareadas.csv',paired)]:
        with (dest/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    matrix=np.array([[100*s['mission_success'] for s in data[name]['summary']] for name in data])
    fig,ax=plt.subplots(figsize=(10.5,5.3),layout='constrained')
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(TEST_LEVELS)),[f'{100*x:g}%' for x in TEST_LEVELS])
    ax.set_yticks(range(len(data)),[LABELS[x] for x in data])
    ax.set_xlabel('Pérdida de RPM en el motor 0\nÉxito: último waypoint a menos de 0,25 m; no certifica contacto con suelo.\nSimulación con falla conocida; una semilla de entrenamiento por arquitectura.',fontsize=9)
    ax.set_title('Rutas completadas: 50 escenarios comunes por severidad')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,f'{matrix[i,j]:.0f}%',ha='center',va='center',
                    color='white' if matrix[i,j]>65 else '#151515',fontsize=11)
    fig.colorbar(im,ax=ax,label='Éxito de misión (%)',fraction=.035,pad=.03)
    fig.savefig(dest/'comparacion_modelos.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.2,5),layout='constrained')
    for i,name in enumerate(data):
        ax.plot(np.array(TEST_LEVELS)*100,matrix[i],marker='o',label=LABELS[name],
           color='black' if name=='baseline' else None,linestyle='--' if name=='baseline' else '-',linewidth=1.6)
    ax.set(xlabel='Pérdida de RPM en el motor 0 (%)',ylabel='Misiones completas (%)',ylim=(-2,102),xticks=np.array(TEST_LEVELS)*100)
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.savefig(dest/'curvas_exito.png',dpi=220);plt.close(fig)
    sources=[]
    paths=sorted(list((ROOT/'experiments').rglob('*.py'))+list((ROOT/'fallas').glob('*.py'))+
                 [ROOT/'comparacion/comparar_base.py'])
    for path in paths:sources.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(dest/'provenance.json',dict(date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      source_commit=commit,local_uncommitted_corrections=True,reward_sha256=reward_hash,code=sources,
      checkpoint_sha256={a:hashlib.sha256((out/a/'best.zip').read_bytes()).hexdigest() for a in ARCHS},
      configs=configs,training_seconds={k:v['training_seconds'] for k,v in training.items()},
      hardware=dict(cpu='Intel Core i9-12900K',gpu='NVIDIA RTX 3080 Ti 12 GB',ram_gb=32)))
    reward=json.loads((out/'reward.json').read_text(encoding='utf-8'))
    hp=json.loads((out/'hyperparameter_selection.json').read_text(encoding='utf-8'))
    lines=['TESIS DE NICOL — CORRECCIONES Y COMPARACIÓN LOCAL','',
      'Código y resultados: '+str(ROOT.parent),
      'Repositorio de origen: https://github.com/YaciraUTEC/PFC/tree/bounded-features-no-sign-clip',
      'Commit de origen: '+commit,'',
      'RESULTADOS (porcentaje de misiones completas; 50 rutas comunes por celda)',
      'Modelo'.ljust(20)+' '.join(f'{100*x:g}%'.rjust(7) for x in TEST_LEVELS)]
    for i,name in enumerate(data):lines.append(LABELS[name].ljust(20)+' '.join(f'{x:.0f}'.rjust(7) for x in matrix[i]))
    lines += ['', 'Las columnas indican pérdida de RPM de un motor, no pérdida de empuje.',
      '10% RPM = 19% empuje; 20% RPM = 36% empuje. Falla persistente en motor 0, a mitad del crucero.',
      'Misiones completas y recuperación (misión o aterrizaje cercano tras exposición) se guardan separadas.',
      'Misión completa significa agotar waypoints con tolerancia de 0.25 m; no certifica contacto físico con el suelo.',
      'Los vuelos planeados con falla que terminan antes de exposición permanecen en el denominador.',
      'CSV incluye intervalos Wilson al 95%; no representan variabilidad entre semillas de entrenamiento.',
      'diferencias_pareadas.csv cuenta rutas mejoradas y empeoradas contra la base para cada severidad.',
      '', 'CORRECCIONES',
      '- Base exacta DSLPIDControl tanto para demostraciones como para el control residual.',
      '- Niveles de falla propagados al entorno; cobertura efectiva hasta 20%.',
      '- Acción coherente entre búsqueda IRL, entrenamiento y evaluación: residual fijo de hasta 3000 RPM, solo tras falla.',
      '- Pesos y escalas viajan juntos; restricción de signos durante aprendizaje, sin recorte posterior de pesos negativos.',
      '- Se elimina el sesgo de imitar residual cero: dos características relativas a la política quedan fuera del ajuste.',
      '- Memoria de características actualizada todo el vuelo; recuperación de altura activa respecto a altura de crucero.',
      '- Semillas reproducibles y evaluación final con rutas explícitamente pareadas.',
      '- Puntos guardados después de actualizar PPO; reanudación comprobada con optimizador y calendario.',
      '', 'ENTRENAMIENTO EJECUTADO',
      '1600 demostraciones DSLPIDControl, nominal y fallas leves 2-10%. Se conservan todos los desenlaces.',
      'IRL: 8 iteraciones x 100000 pasos nominales solicitados (PPO redondea a rollouts completos).',
      f'Iteración de recompensa seleccionada: {reward["best_iteration"]}; distancia de proyección {reward["margin"]:.6f}.',
      'La distancia no alcanzó un criterio que permita afirmar convergencia; la recompensa se congeló por presupuesto.',
      'Recompensa de características aprendida + prioridades de tarea explícitas comunes. No es IRL puro.',
      'Prioridades: misión +100, caída -120, aterrizaje cercano +40, lejano -60, timeout -30, vivo +.02.',
      'Penalización residual explícita .02; estilo IRL x .01.',
      f'Pilotos LSTM 32768 pasos: tasas 1e-4 y 3e-4; selección en validación: {hp["chosen_lr"]:g}.',
      'Las tasas empataron en el piloto; se eligió la menor. No se afirma que sea óptima globalmente.',
      'PPO: 524288 pasos de aprendizaje por arquitectura, semilla 42; historial 32 a 48 Hz.',
      '32 entornos, rollout total 2048, batch 128, hasta 5 épocas; gamma .99, GAE .95, entropía .005.',
      'Clip .15, target KL .015, gradiente .5; tasa lineal hasta 10% de la inicial.',
      'Currículo hasta 20% RPM en el primer 40% del presupuesto. El prefijo físico DSLPID se simula íntegro.',
      'Los pasos de aprendizaje omiten el prefijo sin intervención neuronal. Se verificó igualdad de estados e historial.',
      'Validación desde semilla 50000, con reinicios automáticos por proceso (no emparejamiento exacto de rutas).',
      'Checkpoint elegido en validación antes de evaluar pruebas pareadas desde 70000.',
      '', 'MODELOS Y ALCANCE',
      'LSTM; Mamba-1; Mamba-2; Mamba-3 SISO; GTrXL; LTC de ncps (no CfC), tres subpasos ODE.',
      'Mamba usa referencias PyTorch comprobadas contra ecuaciones/referencias oficiales; GTrXL fuente DI-engine.',
      'LTC compila su misma celda para GPU Windows. No se comparan velocidades con kernels oficiales de Linux.',
      'Extractores de aproximadamente 150000 parámetros, con actor y crítico PPO comunes.',
      'LSTM, Mamba y GTrXL usan dos capas; LTC usa una celda ODE densa de 160 unidades (el argumento layers no se aplica).',
      'Se trata de compensación residual con indicador y severidad de falla conocidos. No evalúa detección autónoma.',
      'El residual se desactiva en vuelos nominales. Sus éxitos nominales pertenecen a DSLPIDControl.',
      'Un solo entrenamiento por arquitectura y 50 rutas por severidad: primera comparación, no ranking definitivo.',
      'No se ensayan otros motores, momentos de falla, dinámica real ni perturbaciones externas.',
      'No se ajustó el presupuesto o la recompensa a partir de los resultados de prueba de una arquitectura.',
      'Falta el checkpoint exacto del resultado histórico de Nicol al 4%; no se reproduce ni se compara directamente.',
      'Las nuevas corridas combinan correcciones, recompensa, hiperparámetros y arquitectura; no aíslan el aporte de cada cambio.',
      'Tampoco incluyen una ablación sin IRL o sin memoria temporal; no demuestran por sí solas el aporte causal de esos componentes.',
      '', 'ARTEFACTOS',
      'resultados/comparacion_modelos.png; curvas_exito.png; comparacion.csv; diferencias_pareadas.csv; provenance.json.',
      'PFC/runs/dslpid_comparison_v2/reward.json: recompensa congelada y escalas.',
      'PFC/runs/dslpid_comparison_v2/<modelo>/best.zip: checkpoint elegido con validación.',
      'PFC/runs/dslpid_comparison_v2/<modelo>/final.zip y latest.zip: estado final y punto reanudable.',
      'PFC/runs/dslpid_comparison_v2/<modelo>/paired_test.json: evaluación definitiva por episodio.',
      'PFC/runs/dslpid_comparison_v2/<modelo>/config.json y validation.json: protocolo y curva.',
      'revision/pruebas.xml: pruebas automáticas; revision/ltc_smoke/result.json: PPO GPU compilado y recarga.',
      'revision/dependencias_instaladas.txt; PFC/experiments/README.txt: versiones y reproducción.',
      'REVISION_TESIS_NICOL.txt conserva el diagnóstico anterior a las correcciones.',
      '', 'No se subió ningún cambio al repositorio remoto. No se modificaron los pesos históricos reportados por Nicol.']
    latency_path=dest/'latencia_decision.json'
    if latency_path.exists():
        latency=json.loads(latency_path.read_text(encoding='utf-8'))
        lines += ['', 'TIEMPO DE DECISIÓN EN ESTA PC (milisegundos; mediana / percentil 95)',
          'Modelo'.ljust(20)+'CPU, 1 hilo'.rjust(21)+'GPU RTX 3080 Ti'.rjust(22)]
        for row in latency['models']:
            cpu,gpu=row['cpu'],row['cuda']
            lines.append(LABELS[row['architecture']].ljust(20)+
                f"{cpu['median_ms']:.2f} / {cpu['p95_ms']:.2f}".rjust(21)+
                f"{gpu['median_ms']:.2f} / {gpu['p95_ms']:.2f}".rjust(22))
        lines += ['Una acción, historial 32; predict determinista, incluye transferencias CPU/GPU.',
          'No incluye sensores, DSLPIDControl ni salida al actuador. No es medida a bordo ni garantía de tiempo real.',
          'Referencia temporal: 48 Hz equivale a 20.83 ms por ciclo. Los tiempos de entrenamiento no son un benchmark aislado.']
    (ROOT.parent/'INFORME_ENTRENAMIENTO_NICOL.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    archive_paths={ROOT/p for p in changed if (ROOT/p).is_file()}
    archive_paths.update((ROOT/'experiments').rglob('*'))
    archive_paths.update([ROOT/'fallas/reward_contract.py',ROOT/'.gitignore'])
    with zipfile.ZipFile(ROOT.parent/'revision/codigo_corregido.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(archive_paths):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path,path.relative_to(ROOT))
    diff=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    (ROOT.parent/'revision/correcciones_tracked.patch').write_bytes(diff)
    save(ROOT.parent/'FINALIZADO.json',dict(status='completed',architectures=ARCHS,
      steps_per_model=524288,paired_scenarios_per_level=50,levels=TEST_LEVELS,
      total_final_test_flights=7*50*len(TEST_LEVELS),reward_sha256=reward_hash,
      report=str(ROOT.parent/'INFORME_ENTRENAMIENTO_NICOL.txt'),
      numerical_tests='revision/pruebas.xml',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('\n'.join(lines[:14]))
if __name__=='__main__':main()