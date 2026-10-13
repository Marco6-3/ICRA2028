"""Reproduce normalization, teacher, students, paired closed-loop and report."""
import argparse,json,os,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--official-repo',type=Path,default=Path('C:/isaac_ws/FlexiTac-IsaacSim-Simulation'))
    a=p.parse_args(); a.out=a.out.resolve(); a.out.mkdir(exist_ok=False,parents=True)
    root=Path(__file__).resolve().parent; environment=os.environ.copy(); environment.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
    status={'status':'running','completed':[]}
    def run(stage,script,args,manifest=False):
        command=[sys.executable,str(root/script),*map(str,args)]
        print('STAGE',stage,flush=True)
        with (a.out/(stage+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,cwd=root.parents[1])
        if result.returncode: raise RuntimeError(f'{stage} exit {result.returncode}; see log')
        if manifest:
            m=json.loads((a.out/stage/'manifest.json').read_text(encoding='utf-8'))
            if m['status']!='completed' or m['errors'] or m.get('rejected_initializations'): raise RuntimeError(f'{stage} incomplete')
        status['completed'].append(stage); (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    try:
        run('bounded_pure','bounded_probe.py',['--source',root/'outputs/pure_loading_01','--out',a.out/'bounded_pure'])
        run('bounded_original','e2_dev.py',['--source',root/'outputs/pilot_tangent_02','--out',a.out/'bounded_original','--profiles','static_mcf','static_s4','mcf_t16','--normalization','robust_floor_clip8','--protocol',root/'REFLEX_PROTOCOL.md'])
        shared=['--official-repo',a.official_repo,'--fixture-npz',root/'outputs/contact_01/table_open/c000_r00.npz','--support','removed','--preload-mode','open','--initial-close-rad','.003','--load-frame','reference','--reference-from-finger','2','--trial-steps','180','--headless','--device','cpu','--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false']
        run('teacher','reflex_sim.py',[*shared,'--out',a.out/'teacher','--matrix',root/'reflex_teacher_matrix.json','--controller','oracle'],True)
        run('students','train_reflex.py',['--source',a.out/'teacher','--out',a.out/'students'])
        for controller in ('nominal','high','mcf','scalar','oracle'):
            ck=['--checkpoint',a.out/'students'/f'{controller}_selected.pt'] if controller in ('mcf','scalar') else []
            run(controller,'reflex_sim.py',[*shared,'--out',a.out/controller,'--matrix',root/'reflex_test_matrix.json','--controller',controller,*ck],True)
        report_args=['--out',a.out/'report','--students',a.out/'students','--bounded-pure',a.out/'bounded_pure']
        for k in ('nominal','high','mcf','scalar','oracle'): report_args+=['--'+k,a.out/k]
        report_args[report_args.index('--out')+1]=a.out/'report_v1'
        run('report_v1','report_reflex.py',report_args)
        run('dagger','reflex_sim.py',[*shared,'--out',a.out/'dagger','--matrix',root/'reflex_dagger_matrix.json','--controller','mcf','--checkpoint',a.out/'students/mcf_selected.pt'],True)
        run('stream_students','train_reflex_stream.py',['--source',a.out/'teacher','--dagger',a.out/'dagger','--out',a.out/'stream_students'])
        for controller in ('nominal','high','mcf','scalar','oracle'):
            ck=['--checkpoint',a.out/'stream_students'/f'{controller}_selected.pt'] if controller in ('mcf','scalar') else []
            run('stream_'+controller,'reflex_sim.py',[*shared,'--out',a.out/('stream_'+controller),'--matrix',root/'reflex_test_stream_matrix.json','--controller',controller,*ck],True)
        report_args=['--out',a.out/'report','--students',a.out/'stream_students','--bounded-pure',a.out/'bounded_pure']
        for k in ('nominal','high','mcf','scalar','oracle'): report_args+=['--'+k,a.out/('stream_'+k)]
        run('report','report_reflex.py',report_args)
        status['status']='completed'
    except Exception as e:
        status.update(status='failed',error=repr(e)); raise
    finally: (a.out/'pipeline.json').write_text(json.dumps(status,indent=2),encoding='utf-8')


if __name__=='__main__': main()
