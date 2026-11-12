"""Scientific figures from completed frozen results; no model selection here."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import OUT
from .inventory import ARCHS,TEST_LEVELS
NAMES=dict(lstm='LSTM',mamba='Mamba',mamba2='Mamba-2',mamba3='Mamba-3 SISO',gtrxl='GTrXL',ltc='LTC',csil='CSIL',meirl='MEIRL',rime='RIME')

def data():
    with (OUT/'success_rates.csv').open(encoding='utf-8') as f:return list(csv.DictReader(f))
def values(rows,kind,name,mode,condition,levels):
    rr={round(float(r['level']),6):float(r['success_percent']) for r in rows if r['kind']==kind and r['name']==name and r['mode']==mode and r['condition']==condition}
    return [rr[round(float(l),6)] for l in levels]
def base(ax,rows,condition,levels):
    for name,label,style in [('original','DSLPID anterior',':'),('observer_only','DSLPID: solo estimador','--'),('selected','DSLPID: perfil completo','-')]:
        ax.plot(np.array(levels)*100,values(rows,'baseline',name,'baseline',condition,levels),style,color='black' if name=='selected' else '.5',lw=2.3,label=label)
    ax.set(xlabel='Pérdida de RPM de un motor (%)',ylabel='Aterrizajes controlados y estables (%)',ylim=(-3,103));ax.grid(alpha=.2)
def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close(fig)
def curves(rows,kind,names,filename,title):
    levels=TEST_LEVELS;fig,axes=plt.subplots(1,2,figsize=(13.5,5),sharey=True)
    for ax,mode,caption in zip(axes,['raw','selected'],['Compensación aprendida sin filtro externo','Estrategia elegida solo en validación']):
        base(ax,rows,'estimated',levels)
        for name in names:ax.plot(np.array(levels)*100,values(rows,kind,name,mode,'estimated',levels),marker='.',label=NAMES[name],lw=1.4)
        if kind=='reward':
            for mode0,label,style in [('bc','LSTM solo imitación','--'),('raw','LSTM PPO manual','-.')]:ax.plot(np.array(levels)*100,values(rows,'residual','lstm',mode0,'estimated',levels),style,label=label,lw=1.5)
        ax.set_title(caption,fontsize=10);ax.legend(fontsize=8,loc='lower left')
    fig.suptitle(title+' | 5 semillas × 60 rutas por nivel; efectividad estimada',fontsize=11)
    fig.subplots_adjust(top=.84);finish(fig,filename)
def main():
    rows=data();curves(rows,'residual',ARCHS,'curvas_residuales.png','Corrección directa de motores')
    curves(rows,'observer',ARCHS,'curvas_estimadores.png','Corrección aprendida de efectividad')
    curves(rows,'reward',['csil','meirl','rime'],'curvas_recompensas.png','Recompensas adaptadas sobre LSTM')
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,condition in zip(axes,['estimated','exact']):
        base(ax,rows,condition,TEST_LEVELS)
        for name in ARCHS:ax.plot(np.array(TEST_LEVELS)*100,values(rows,'residual',name,'raw',condition,TEST_LEVELS),'.-',label=NAMES[name],lw=1.2)
        ax.set_title('Efectividad '+('estimada' if condition=='estimated' else 'exacta'));ax.legend(fontsize=8,loc='lower left')
    finish(fig,'efectividad_exacta_y_estimada.png')
    family=json.loads((OUT/'family_simultaneous_intervals.json').read_text())['contrasts'];fig,ax=plt.subplots(figsize=(9,7))
    for index,row in enumerate(family):
        point=100*row['mean'];ax.errorbar(point,index,xerr=[[point-100*row['lower']],[100*row['upper']-point]],fmt='o',color='tab:blue',capsize=3)
    ax.set_yticks(range(len(family)),[r['group'] for r in family]);ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.2)
    ax.set(xlabel='Ganancia frente a DSLPID actualizado (puntos porcentuales)',title='28–29.5% estimada | intervalos simultáneos de las 15 familias');finish(fig,'intervalos_familias.png')
    decision=json.loads((OUT/'decision.json').read_text());manifest=json.loads((OUT/'confirmation_manifest.json').read_text());fig,ax=plt.subplots(figsize=(10,4.7));labels=[];success=[]
    for spec in manifest['specifications']:
        rr=json.loads((OUT/'confirmation'/f"{spec['id']}.json").read_text());labels.append(spec['id']);success.append(100*np.mean([r['stable_landing'] for r in rr if r['role']=='primary']))
    bars=ax.bar(labels,success,color=['tab:green' if l==decision['recommended'] else 'tab:blue' for l in labels]);ax.bar_label(bars,fmt='%.2f%%');ax.set(ylim=(0,110),ylabel='Aterrizajes estables (%)',title='Confirmación independiente: 1000 rutas × 3 fallas severas por controlador');ax.tick_params(axis='x',labelrotation=12);finish(fig,'confirmacion_independiente.png')
    labels=['original','selected','balanced_governor','ready_only','observer_only','yaw_only','reserve_only','without_observer','without_ready','without_yaw','without_reserve']
    grid=np.array([values(rows,'baseline',name,'baseline','estimated',[.25,.28,.29,.295,.296,.30,.35,.50]) for name in labels]);fig,ax=plt.subplots(figsize=(10,7));image=ax.imshow(grid,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set_xticks(range(8),['25','28','29','29.5','29.6','30','35','50']);ax.set_yticks(range(len(labels)),labels);ax.set(xlabel='Pérdida de RPM (%)',title='Ablaciones estructuradas | 60 rutas por celda, efectividad estimada')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):ax.text(j,i,f'{grid[i,j]:.1f}',ha='center',va='center',fontsize=8,color='black' if grid[i,j]>55 else 'white')
    fig.colorbar(image,ax=ax,label='Aterrizajes estables (%)');finish(fig,'ablaciones_y_limites.png')
if __name__=='__main__':main()
