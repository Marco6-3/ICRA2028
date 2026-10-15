"""Upload independent verified archives to the already-created draft Release."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3); p.add_argument('--direct',action='store_true'); a=p.parse_args()
    assert 1<=a.workers<=48
    index=json.loads(Path(__file__).with_name('raw_assets.json').read_text()); gh=shutil.which('gh'); assert gh
    env=os.environ.copy()
    if a.direct:
        env={k:v for k,v in env.items() if k.upper() not in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY')}
        env['NO_PROXY']='*'
    def release():
        rows=json.loads(subprocess.check_output([gh,'api',f"repos/{index['repository']}/releases?per_page=100"],env=env))
        return next(row for row in rows if row['tag_name']==index['tag'])
    current=release(); endpoint=f"repos/{index['repository']}/releases/{current['id']}"
    remote={asset['name']:asset for asset in current['assets']}
    def upload(asset):
        path=a.archive_dir/asset['name']
        assert path.stat().st_size==asset['bytes']
        with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==asset['sha256']
        if remote.get(asset['name'],{}).get('digest')=='sha256:'+asset['sha256']: return asset['name']+' (verified existing)'
        for attempt in range(3):
            result=subprocess.run([gh,'release','upload',index['tag'],str(path.resolve()),'--repo',index['repository']],capture_output=True,text=True,env=env)
            if result.returncode==0: return asset['name']
            # An uncertain response may already have uploaded the exact bytes.
            check=subprocess.run([gh,'api',endpoint],capture_output=True,text=True,env=env)
            if check.returncode==0:
                matches=[v for v in json.loads(check.stdout)['assets'] if v['name']==asset['name']]
                if matches and matches[0].get('digest')=='sha256:'+asset['sha256']: return asset['name']
            if attempt==2: raise RuntimeError(asset['name']+': '+result.stderr)
            time.sleep(2*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(upload,asset) for asset in index['assets']]
        for completed,future in enumerate(concurrent.futures.as_completed(futures),1):
            print('UPLOADED',completed,'/',len(futures),future.result(),flush=True)
    print('ALL_ASSETS_UPLOADED',len(index['assets']),flush=True)

if __name__=='__main__': main()
