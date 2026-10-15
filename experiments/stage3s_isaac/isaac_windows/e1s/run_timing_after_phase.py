import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--phase',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
assert not a.out.exists()
start=time.monotonic()
while True:
    state=json.loads((a.phase/'pipeline.json').read_text()) if (a.phase/'pipeline.json').exists() else {'status':'running'}
    if state['status']=='failed': raise RuntimeError('Main phase scan failed; do not overlap Isaac apps')
    if state['status']=='completed': break
    if time.monotonic()-start>1800: raise RuntimeError('Main phase scan exceeded queue wait budget')
    time.sleep(3)
env=os.environ.copy(); env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
command=[sys.executable,str(ROOT/'timing_intervention_sim.py'),'--out',str(a.out.resolve()),'--matrix',str(ROOT/'timing_intervention_matrix.json'),
    '--fixture-npz',str(ROOT/'outputs/contact_01/table_open/c000_r00.npz'),'--support','removed','--preload-mode','open','--initial-close-rad','.003',
    '--load-frame','reference','--reference-from-finger','2','--trial-steps','180','--controller','timed','--headless','--device','cpu',
    '--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false']
print('TIMING_INTERVENTION_START',flush=True)
result=subprocess.run(command,env=env)
manifest=json.loads((a.out/'manifest.json').read_text())
assert result.returncode==0 and manifest['status']=='completed' and not manifest['errors']
print('TIMING_INTERVENTION_COMPLETED',len(manifest['completed_trials']),flush=True)
