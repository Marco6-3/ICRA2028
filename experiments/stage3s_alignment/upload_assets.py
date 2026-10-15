"""Upload independent verified archives to the already-created draft Release."""
import argparse,concurrent.futures,hashlib,json,shutil,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive-dir',type=Path,required=True); a=p.parse_args()
    index=json.loads(Path(__file__).with_name('raw_assets.json').read_text()); gh=shutil.which('gh'); assert gh
    def upload(asset):
        path=a.archive_dir/asset['name']
        assert path.stat().st_size==asset['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256']
        for attempt in range(3):
            result=subprocess.run([gh,'release','upload',index['tag'],str(path.resolve()),'--repo',index['repository']],capture_output=True,text=True)
            if result.returncode==0: return asset['name']
            # An uncertain response may already have uploaded the exact bytes.
            remote=subprocess.run([gh,'api',f"repos/{index['repository']}/releases/tags/{index['tag']}"],capture_output=True,text=True)
            if remote.returncode==0:
                matches=[v for v in json.loads(remote.stdout)['assets'] if v['name']==asset['name']]
                if matches and matches[0].get('digest')=='sha256:'+asset['sha256']: return asset['name']
            if attempt==2: raise RuntimeError(asset['name']+': '+result.stderr)
            time.sleep(2*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures=[executor.submit(upload,asset) for asset in index['assets']]
        for completed,future in enumerate(concurrent.futures.as_completed(futures),1):
            print('UPLOADED',completed,'/',len(futures),future.result(),flush=True)
    print('ALL_ASSETS_UPLOADED',len(index['assets']),flush=True)

if __name__=='__main__': main()
