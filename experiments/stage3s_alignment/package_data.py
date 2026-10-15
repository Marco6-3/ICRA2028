"""Archive new native traces, preserving NPZ bytes and member digests."""
import argparse, hashlib, json, zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'stage3s_isaac'
E1=BASE/'isaac_windows/e1s'
TAG='stage3s-phase-alignment-20261014'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--asset-dir',type=Path,required=True); a=p.parse_args()
    assert json.loads((E1/'outputs/alignment_report_20261014_01/verification.json').read_text())['status']=='passed'
    assert json.loads((E1/'outputs/phase_report_20261014_01/verification.json').read_text())['status']=='passed'
    a.asset_dir.mkdir(parents=True,exist_ok=False)
    folders=['phase_20261014_01','timing_intervention_20261014_01','alignment_20261014_01','alignment_smoke_20261014_01','trigger_static_reference_20261014_01']
    groups=[]
    for folder in folders:
        directory=E1/'outputs'/folder
        if (directory/'manifest.json').exists(): groups.append((folder,directory))
        else: groups.extend((folder+'_'+x.name,x) for x in sorted(directory.iterdir()) if x.is_dir() and (x/'manifest.json').exists())
    assets=[]; files={}; part=0
    for group,directory in groups:
        manifest=json.loads((directory/'manifest.json').read_text())
        paths=sorted(directory.glob('*.npz')); assert paths
        trace_names=[]
        for tid in manifest['completed_trials']:
            for suffix in ('.json','_initialization.json'):
                path=directory/(tid+suffix); assert path.exists(); paths.append(path); trace_names.append(path.name)
        # Keep aggregate tables / provenance reviewable in Git; archive every
        # per-trial JSON beside its unchanged NPZ instead of a 4000-file PR.
        (directory/'.gitignore').write_text('*.npz\n'+''.join('/'+name+'\n' for name in sorted(trace_names)),encoding='utf-8')
        batches=[]; current=[]; size=0
        for path in paths:
            if current and size+path.stat().st_size>160_000_000:
                batches.append(current); current=[]; size=0
            current.append(path); size+=path.stat().st_size
        if current: batches.append(current)
        for batch in batches:
            part+=1; name=f'phase-alignment-20261014-part{part:02d}.zip'; archive=a.asset_dir/name
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
                for path in batch:
                    member=path.relative_to(BASE).as_posix()
                    z.write(path,member)
                    files[member]=dict(sha256=sha(path),bytes=path.stat().st_size,asset=name,group=group)
            # Verify archived bytes directly, independently of the original path.
            with zipfile.ZipFile(archive) as z:
                assert len(z.namelist())==len(batch)
                for member in z.namelist(): assert hashlib.sha256(z.read(member)).hexdigest()==files[member]['sha256']
            assets.append(dict(name=name,sha256=sha(archive),bytes=archive.stat().st_size,members=len(batch)))
            print('PACKED',name,len(batch),flush=True)
    result=dict(tag=TAG,repository='Marco6-3/ICRA2028',assets=assets,files=files,total_files=len(files),
        total_raw_bytes=sum(x['bytes'] for x in files.values()),format='Unmodified native NPZ and per-trial JSON; ZIP_STORED; relative to experiments/stage3s_isaac')
    result['raw_npz_files']=sum(name.endswith('.npz') for name in files)
    result['trial_json_files']=sum(name.endswith('.json') for name in files)
    assert result['raw_npz_files']==1868 and result['trial_json_files']==3686
    for member,entry in files.items():
        if member.endswith('.json') and not member.endswith('_initialization.json'):
            meta=json.loads((BASE/member).read_text(encoding='utf-8'))
            assert files[member[:-5]+'.npz']['sha256']==meta['npz_sha256']
    (HERE/'raw_assets.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ARCHIVED',len(files),result['total_raw_bytes'],'bytes')

if __name__=='__main__': main()
