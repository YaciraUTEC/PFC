"""Data acquisition and one GPU trainer; durable completion per stage and tier."""
import sys,time,subprocess
from .common import OUT,ROOT,read,save,close_milestone
from .power import awake

def launch(name,args):
    handle=(OUT/(name+'.log')).open('a',encoding='utf-8');handle.write('\nATTEMPT '+str(time.time())+'\n');handle.flush()
    p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    save(OUT/(name+'_process.json'),dict(pid=p.pid,started_epoch=time.time(),args=args));return p,handle

def main():
    if not (OUT/'h1_pilot_data_conclusion.json').exists():raise RuntimeError('Close the pilot first')
    started=time.time();allfits=[]
    with awake():
        data,log=launch('data_256',['-m','experiments.round7.data','--groups','256','--workers','8'])
        try:
            for groups in [16,64,256]:
                while not all((OUT/'demonstrations'/f'group_{i:03d}.json').exists() for i in range(groups)):
                    if data.poll() is not None and data.returncode:raise RuntimeError('Data collection failed')
                    save(OUT/'status.json',dict(stage='h1_data',waiting_for_groups=groups,collected=len(list((OUT/'demonstrations').glob('*.json')))));time.sleep(5)
                p,f=launch(f'bc_{groups}',['-m','experiments.round7.bc','--groups',str(groups)])
                save(OUT/'status.json',dict(stage='h1_bc',groups=groups,pid=p.pid))
                try:
                    while p.poll() is None:
                        if data.poll() is not None and data.returncode:p.terminate();raise RuntimeError('Data collection failed during BC')
                        time.sleep(5)
                    if p.returncode:raise RuntimeError(f'BC {groups} failed')
                finally:f.close()
                fits=[read(OUT/'bc_curve'/f'groups_{groups}'/f'seed_{s}'/'complete.json') for s in [301,302,303]];allfits+=fits
                close_milestone(f'h1_scale_{groups}','Matched recurrent clones fitted; reserved flight validation measured before advancing.',fits=fits)
            data.wait()
            if data.returncode:raise RuntimeError('Data collection failed')
        finally:
            log.close()
            if data.poll() is None:data.terminate();data.wait()
        means={g:sum(r['best_success'] for r in allfits if r['groups']==g)/3 for g in [16,64,256]};winner=max(means,key=lambda g:(means[g],-g))
        close_milestone('h1_data_and_proposals','Proposal-data scaling completed; selected using reserved flights, not final test.',means=means,selected_groups=winner,results=allfits,seconds=time.time()-started)
        save(OUT/'status.json',dict(stage='h1_closed',status='complete',selected_groups=winner))
if __name__=='__main__':main()
