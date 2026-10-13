"""Restore exact physical NPZ bytes from hash-verified repository Release assets."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive-dir',type=Path,default=ROOT/'release_cache'); p.add_argument('--download',action='store_true')
    a=p.parse_args(); spec=json.loads((ROOT/'release_assets.json').read_text(encoding='utf-8'))
    rows=json.loads((ROOT/'raw_data_manifest.json').read_text(encoding='utf-8'))['files']; a.archive_dir.mkdir(exist_ok=True,parents=True)
    if a.download:
        subprocess.run(['gh','release','download',spec['tag'],'--repo','Marco6-3/ICRA2028','--pattern','*20261013*.zip','--dir',str(a.archive_dir),'--skip-existing'],check=True)
    count=0
    for asset in spec['assets']:
        path=a.archive_dir/asset['name']
        if path.stat().st_size!=asset['size_bytes'] or sha(path)!=asset['sha256']: raise RuntimeError('Archive hash failed: '+asset['name'])
        allowed={r['path']:r for r in rows if r['asset']==asset['name']}
        with zipfile.ZipFile(path) as z:
            if set(z.namelist())!=set(allowed): raise RuntimeError('Archive file list differs from manifest')
            for name,meta in allowed.items():
                relative=PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts: raise RuntimeError('Unsafe archive path')
                target=(ROOT/Path(*relative.parts)).resolve()
                if not target.is_relative_to(ROOT): raise RuntimeError('Archive path escapes study directory')
                data=z.read(name)
                if len(data)!=meta['size_bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']: raise RuntimeError('NPZ hash failed: '+name)
                if target.exists():
                    if sha(target)!=meta['sha256']: raise FileExistsError('Preserve differing local data: '+str(target))
                else:
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                count+=1
        print('RESTORED',asset['name'],len(allowed),'NPZ',flush=True)
    print('RAW_DATA_VERIFIED',count,flush=True)

if __name__=='__main__': main()
