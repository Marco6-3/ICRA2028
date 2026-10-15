import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out=a.out.resolve(); a.out.mkdir(exist_ok=False,parents=True)
    root=Path(__file__).resolve().parent; env=os.environ.copy(); env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
    shared=['--fixture-npz',root/'outputs/contact_01/table_open/c000_r00.npz','--support','removed','--preload-mode','open','--initial-close-rad','.003','--load-frame','reference','--reference-from-finger','2','--trial-steps','180','--headless','--device','cpu','--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false']
    stages=[('mechanism','nominal','mechanism.json'),('pulse','pulse','pulse.json')]+[(k,k,'phase_all.json') for k in ('nominal','high','mcf','scalar')]+[('oracle_fx','oracle','phase_oracle_fx.json')]
    status={'status':'running','completed':[]}
    try:
        for name,controller,matrix in stages:
            ck=['--checkpoint',root/'outputs/reflex_students_stream_01'/f'{controller}_selected.pt'] if controller in ('mcf','scalar') else []
            command=[sys.executable,str(root/'phase_sim.py'),*map(str,shared),'--out',str(a.out/name),'--matrix',str(root/'phase_matrices'/matrix),'--controller',controller,*map(str,ck)]
            print('PHASE_START',name,flush=True)
            with (a.out/(name+'.log')).open('w',encoding='utf-8') as f: result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
            manifest=json.loads((a.out/name/'manifest.json').read_text(encoding='utf-8'))
            if result.returncode or manifest['status']!='completed' or manifest['errors'] or manifest.get('rejected_initializations'): raise RuntimeError(name+' failed: '+str(manifest))
            status['completed'].append(name); (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            print('PHASE_DONE',name,len(manifest['completed_trials']),flush=True)
        status['status']='completed'
    except Exception as e: status.update(status='failed',error=repr(e)); raise
    finally: (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

if __name__=='__main__': main()
