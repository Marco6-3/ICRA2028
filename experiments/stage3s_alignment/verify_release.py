"""Compare uploaded GitHub asset digests with the local archive index."""
import json,shutil,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
repo=here.parents[1]
gh=shutil.which('gh'); assert gh
def api(path): return json.loads(subprocess.check_output([gh,'api',path]))
index=json.loads((here/'raw_assets.json').read_text())
release=api(f"repos/{index['repository']}/releases/tags/{index['tag']}")
remote={asset['name']:asset for asset in release['assets']}
assert set(remote)=={asset['name'] for asset in index['assets']}
checked=[]
for asset in index['assets']:
    actual=remote[asset['name']]
    assert actual['state']=='uploaded' and actual['size']==asset['bytes']
    assert actual['digest']=='sha256:'+asset['sha256'],asset['name']
    checked.append(dict(name=asset['name'],bytes=actual['size'],server_digest=actual['digest']))
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
tag=api(f"repos/{index['repository']}/git/ref/tags/{index['tag']}")['object']
if tag['type']=='tag': tag=api(f"repos/{index['repository']}/git/tags/{tag['sha']}")['object']
assert tag['type']=='commit' and tag['sha']==head
receipt=dict(status='passed',commit=head,tag=index['tag'],url=release['html_url'],assets=checked,
    total_files=index['total_files'],raw_npz_files=index['raw_npz_files'],trial_json_files=index['trial_json_files'],
    original_bytes_verified_in_local_archives=True,github_server_digests_match=True)
(repo.parent/'alignment_release_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('RELEASE_VERIFIED',len(checked),'assets',head)
