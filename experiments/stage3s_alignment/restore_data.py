"""Verify and restore the additional phase/alignment traces without overwriting."""
import argparse, hashlib, json, subprocess, zipfile
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
BASE=(HERE.parent/'stage3s_isaac').resolve()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive-dir',type=Path,default=HERE/'.download'); p.add_argument('--download',action='store_true'); a=p.parse_args()
    index=json.loads((HERE/'raw_assets.json').read_text()); a.archive_dir.mkdir(parents=True,exist_ok=True)
    count=0
    for asset in index['assets']:
        archive=a.archive_dir/asset['name']
        if not archive.exists() and a.download:
            subprocess.run(['gh','release','download',index['tag'],'--repo',index['repository'],'--pattern',asset['name'],'--dir',str(a.archive_dir)],check=True)
        assert archive.stat().st_size==asset['bytes'] and sha(archive)==asset['sha256'],archive
        with zipfile.ZipFile(archive) as z:
            members=z.namelist(); expected=[name for name,entry in index['files'].items() if entry['asset']==asset['name']]
            assert sorted(members)==sorted(expected)
            for member in members:
                path=PurePosixPath(member)
                assert not path.is_absolute() and '..' not in path.parts and ':' not in member
                target=(BASE/Path(*path.parts)).resolve(); assert target.is_relative_to(BASE)
                entry=index['files'][member]; data=z.read(member)
                assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
                if target.exists(): assert sha(target)==entry['sha256'],'Existing different file: '+str(target)
                else: target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                count+=1
    assert count==index['total_files']; print('RESTORED_AND_VERIFIED',count)

if __name__=='__main__': main()
