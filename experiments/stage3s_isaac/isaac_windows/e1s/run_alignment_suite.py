"""Sequential native preload-alignment and trigger interventions."""
import argparse, json, os, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    a.out=a.out.resolve(); a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'.gitignore').write_text('**/*.npz\n',encoding='utf-8')
    env=os.environ.copy(); env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
    shared=['--fixture-npz',root/'outputs/contact_01/table_open/c000_r00.npz','--support','removed',
        '--initial-close-rad','.003','--load-frame','reference','--trial-steps','180',
        '--reference-file',root/'outputs/phase_20261014_01/nominal/provenance.json',
        '--headless','--device','cpu','--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false']
    stages=[]
    if a.smoke:
        stages=[('low_mcf','mcf',3.06,'none','smoke'),('high_scalar','scalar',4.42,'none','smoke'),
                ('max_fixed','fixed',5.07,'none','smoke')]
    else:
        for level,force in [('low',3.06),('high',4.42)]:
            for kind in ['fixed','scalar','mcf']:
                stages.append((f'{level}_{kind}',kind,force,'none','alignment'))
        stages.append(('max_fixed','fixed',5.07,'none','alignment'))
        for level,force in [('low',3.06),('high',4.42)]:
            for trigger in ['none','pulse','latch']:
                stages.append((f'trigger_{level}_{trigger}','mcf',force,trigger,'trigger'))
    status={'status':'running','stages':[dict(name=n,controller=k,preload_n=f,trigger=t,matrix=m) for n,k,f,t,m in stages],'completed':[]}
    (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    try:
        for name,kind,force,trigger,matrix in stages:
            command=[sys.executable,str(root/'aligned_sim.py'),*map(str,shared),'--controller',kind,
                '--aligned-total-n',str(force),'--trigger',trigger,'--out',str(a.out/name),
                '--matrix',str(root/'alignment_matrices'/f'{matrix}.json')]
            if kind!='fixed': command+=['--checkpoint',str(root/'outputs/reflex_students_stream_01'/f'{kind}_selected.pt')]
            print('ALIGNMENT_START',name,flush=True)
            with (a.out/(name+'.log')).open('w',encoding='utf-8') as f:
                result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
            manifest=json.loads((a.out/name/'manifest.json').read_text(encoding='utf-8'))
            if result.returncode or manifest['status']!='completed' or manifest['errors'] or manifest.get('rejected_initializations'):
                raise RuntimeError(name+' failed: '+str(manifest))
            expected=len(json.loads((root/'alignment_matrices'/f'{matrix}.json').read_text()))
            assert len(manifest['completed_trials'])==expected
            status['completed'].append(name)
            (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
            print('ALIGNMENT_DONE',name,expected,flush=True)
        status['status']='completed'
    except Exception as exc:
        status.update(status='failed',error=repr(exc)); raise
    finally:
        (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

if __name__=='__main__': main()
