"""Verify the exact staged blobs, frozen reports and exclusion of raw traces."""
import hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def git(*args): return subprocess.check_output(['git','-C',str(REPO),*args])
def sha(data): return hashlib.sha256(data).hexdigest()

paths=[p.decode('utf-8') for p in git('diff','--cached','--name-only','-z').split(b'\0') if p]
assert paths and len(paths)<3000,'Keep per-trial traces in hash-verified Release assets.'
assert not git('diff','--cached','--diff-filter=MD','--name-only').strip(),'Historical tracked bytes were modified.'
assert all(p.startswith(('experiments/stage3s_isaac/isaac_windows/e1s/','experiments/stage3s_alignment/')) for p in paths)
assert not any(p.endswith(('.npz','.zip','.pyc')) or '__pycache__' in p for p in paths)
index={}
for item in git('ls-files','-s','-z').split(b'\0'):
    if item:
        left,path=item.split(b'\t',1); index[path.decode('utf-8')]=left.split()[1].decode()
process=subprocess.Popen(['git','-C',str(REPO),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
expected={}
for path in paths:
    process.stdin.write((index[path]+'\n').encode()); process.stdin.flush()
    head=process.stdout.readline().split(); assert head[1]==b'blob'
    data=process.stdout.read(int(head[2])); assert process.stdout.read(1)==b'\n'
    actual=(REPO/path).read_bytes(); assert data==actual,('Staged bytes differ',path)
    if path.endswith('README.md'):
        text=data.decode('utf-8'); assert any('\u4e00'<=c<='\u9fff' for c in text) and '\ufffd' not in text and '??' not in text
    expected[path]=dict(sha256=sha(data),bytes=len(data))
process.stdin.close(); assert process.wait()==0
receipt=dict(status='passed',staged_files=len(paths),staged_bytes=sum(x['bytes'] for x in expected.values()),
    historical_files_unchanged=True,raw_traces_in_release_only=True,exact_staged_blob_bytes=True,files=expected)
(REPO.parent/'alignment_staged_publication_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('STAGED_PUBLICATION_VERIFIED',len(paths),receipt['staged_bytes'],'bytes')
