"""Run the preregistered fixed high-grip reference on the shifted trigger matrix."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
main=root/'outputs/alignment_20261014_01/pipeline.json'
assert json.loads(main.read_text())['status']=='completed'
out=root/'outputs/trigger_static_reference_20261014_01'
assert not out.exists()
freeze={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
    ('TRIGGER_STATIC_REFERENCE_PROTOCOL.md','run_trigger_static_reference.py','alignment_matrices/trigger.json','aligned_sim.py','aligned_policy.py')}
assert freeze==json.loads((root/'trigger_static_reference_freeze.json').read_text())
env=os.environ.copy(); env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1')
command=[sys.executable,str(root/'aligned_sim.py'),'--out',str(out),'--controller','fixed',
    '--aligned-total-n','5.07','--matrix',str(root/'alignment_matrices/trigger.json'),
    '--fixture-npz',str(root/'outputs/contact_01/table_open/c000_r00.npz'),'--support','removed',
    '--initial-close-rad','.003','--load-frame','reference','--trial-steps','180',
    '--reference-file',str(root/'outputs/phase_20261014_01/nominal/provenance.json'),
    '--headless','--device','cpu','--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false']
with out.with_suffix('.log').open('w',encoding='utf-8') as f:
    result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
manifest=json.loads((out/'manifest.json').read_text())
assert result.returncode==0 and manifest['status']=='completed' and not manifest['errors']
assert len(manifest['completed_trials'])==27 and not manifest['rejected_initializations']
(out/'.gitignore').write_text('*.npz\n',encoding='utf-8')
print('TRIGGER_STATIC_REFERENCE_DONE',27)
