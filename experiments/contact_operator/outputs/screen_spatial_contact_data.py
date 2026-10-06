"""Screen real TaF records for load-matched spatial-wrench variation.
No generated pressure maps, no BC, no VLA and no labels derived from 11D.
Run: python outputs/screen_spatial_contact_data.py
Dependencies: numpy, scipy, pyarrow, matplotlib.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'work'/'taf'
OUT=ROOT/'outputs'/'spatial_contact_screen'
CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
BASE='https://huggingface.co/datasets/jiamig/taf-dataset'
REV='239c8ee8156bf11eb6d6c11c004f9a8721528d9d'


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=25) as r:return r.read()
        except OSError:
            if attempt==2:raise


def record(object_id):
    prefix=f'taf_dataset/gs_mini/gs_mini_obj{object_id}/'
    directory=prefix+'data/chunk-000'
    listing_path=CACHE/f'obj{object_id}_listing.json'
    if not listing_path.exists():
        listing_path.write_bytes(fetch('https://huggingface.co/api/datasets/jiamig/taf-dataset/tree/'+REV+'/'+directory))
    entry=next(e for e in json.loads(listing_path.read_text()) if e['type']=='file' and e['path'].endswith('.parquet'))
    path=CACHE/f'obj{object_id}.parquet'
    if not path.exists():path.write_bytes(fetch(BASE+'/resolve/'+REV+'/'+entry['path']))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==entry['lfs']['oid']
    t=pq.read_table(path)
    raw=np.asarray(t['observation.pressure_matrix'].to_pylist(),dtype=np.float32)
    ft=np.asarray(t['observation.force_torque'].to_pylist(),dtype=np.float64)
    ep=np.asarray(t['episode_index']);frame=np.asarray(t['frame_index'])
    assert raw.shape[1:]==(12,12) and np.isfinite(raw).all() and np.isfinite(ft).all()
    assert (raw>=0).all()
    # Initial independent ATI reading supports a contact-free calibration prefix.
    # Do not search future frames to improve calibration or fabricate a baseline.
    calibration_ok=bool(np.max(abs(ft[:30,2]))<.5)
    meta=dict(source_sequence=f'gs_mini_obj{object_id}',path=entry['path'],sha256=digest,
        published_sha256_match=True,rows=len(raw),converted_episode_count=len(np.unique(ep)),
        converted_episode_lengths=[int(np.sum(ep==e)) for e in np.unique(ep)],
        calibration_prefix_frames=30,calibration_prefix_max_abs_Fz=float(np.max(abs(ft[:30,2]))),
        calibration_ok=calibration_ok)
    if not calibration_ok:return None,meta
    baseline=np.median(raw[:30],axis=0)
    mad=np.median(abs(raw[:30]-baseline),axis=0)
    threshold=3*1.4826*mad
    delta=raw-baseline
    weights=np.where(delta>threshold,delta,0).astype(np.float32)
    # Independent wrench reference acquired in the same calibration prefix.
    ft=ft-np.median(ft[:30],axis=0)
    p=weights.sum((1,2));fz=ft[:,2]
    normal=abs(fz)
    with np.errstate(divide='ignore',invalid='ignore'):
        shear=np.linalg.norm(ft[:,:2],axis=1)/normal
        lever=np.c_[-ft[:,4]/fz,ft[:,3]/fz]
    xx,yy=np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,12))
    denom=np.where(p>0,p,1)
    centroid=np.c_[weights.reshape(-1,144)@xx.ravel()/denom,
                   weights.reshape(-1,144)@yy.ravel()/denom]
    arrays=dict(weights=weights,ft=ft,pressure_sum=p,normal=normal,lever=lever,
        centroid=centroid,shear_ratio=shear,episode=ep,frame=frame,
        original_row=np.arange(len(p)),group=np.full(len(p),object_id))
    # Non-overlapping 15-row windows stay inside each converted segment.
    qualified=[]
    for e in np.unique(ep):
        indices=np.flatnonzero(ep==e)
        indices=indices[indices>=30]
        for start in range(0,len(indices)-14,15):
            ix=indices[start:start+15]
            if np.min(normal[ix])<2 or np.min(p[ix])<=0:continue
            p_cv=float(np.std(p[ix])/np.mean(p[ix]));f_cv=float(np.std(normal[ix])/np.mean(normal[ix]))
            if p_cv>.05 or f_cv>.05:continue
            # Wrench variation criterion is independent of pressure centroid.
            label_span=float(np.linalg.norm(np.ptp(lever[ix],axis=0)))
            if label_span<.002:continue
            qualified.append(dict(source_sequence=meta['source_sequence'],episode=int(e),
                start_row=int(ix[0]),end_row=int(ix[-1]),pressure_cv=p_cv,normal_cv=f_cv,
                lever_span_stored_units=label_span,
                centroid_span_normalized=float(np.linalg.norm(np.ptp(centroid[ix],axis=0))),
                max_shear_ratio=float(np.max(shear[ix]))))
    meta['stable_window_count']=len(qualified)
    meta['stable_windows_low_shear_count']=sum(w['max_shear_ratio']<=.2 for w in qualified)
    meta['stable_windows']=qualified
    # Retain one row per 15 frames for pair retrieval, not independent trials.
    mask=(np.arange(len(p))%15==0)&(np.arange(len(p))>=30)&(normal>=2)&(p>0)&(shear<=.2)&np.isfinite(lever).all(1)
    return {k:v[mask] for k,v in arrays.items()},meta


def main():
    protocol=dict(date='2026-10-06',dataset=BASE,revision=REV,source_sequences=[1,2,3,4,5,6],
        purpose='Dataset eligibility and real-pair curation only; no learned representation comparison',
        calibration='First 30 chronological rows only, must have all |Fz|<0.5 in stored units; per-taxel median and 3*1.4826*MAD cutoff; freeze thereafter',
        window='15 rows; pressure CV<=5%, independent |Fz| CV<=5%, |Fz|>=2, external moment/force vector span>=0.002',
        pairs='Pressure-sum ratio and independent-normal-load ratio each within 5%; shear/normal<=0.2; external moment/normal vector difference>=0.003',
        pair_selection='No centroid, orientation or full11D difference enters pair selection',
        external_label='[-Ty/Fz, Tx/Fz] from independent ATI, baseline-subtracted; moment/normal-load ratio only, not verified physical CoP',
        limitations=['No independent jamming, tipping, release or recovery-action labels.',
          'External ATI pose/offset and exact matrix force calibration not verified.',
          'Moment/normal-force ratio can reflect shear moments, sensor offsets or normal-contact redistribution.',
          'Converted episodes can be fixed-length chunks of one recording; split future models by source sequence, not chunks.',
          'Six source sequences were chosen by fixed numeric IDs before reading results; not a formal Stage1 pass.',
          'Different sensor from FlexiTac; do not infer identical noise or physical calibration.'])
    save('protocol.json',protocol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(record,[1,2,3,4,5,6]))
    save('sequence_audit.json',[meta for _,meta in records])
    valid=[a for a,_ in records if a is not None]
    assert valid,'No verified contact-free calibration prefixes'
    arrays={k:np.concatenate([a[k] for a in valid]) for k in valid[0]}
    pos=np.c_[np.log(arrays['pressure_sum']),np.log(arrays['normal'])]
    tree=cKDTree(pos)
    pairs=[];seen=set();radius=np.log(1.05)
    for i in range(len(pos)):
        neighbors=tree.query_ball_point(pos[i],radius,p=np.inf)
        candidates=[j for j in neighbors if j>i and (arrays['group'][i]!=arrays['group'][j] or
                     abs(int(arrays['original_row'][i])-int(arrays['original_row'][j]))>=30)]
        if not candidates:continue
        j=max(candidates,key=lambda j:np.linalg.norm(arrays['lever'][i]-arrays['lever'][j]))
        distance=float(np.linalg.norm(arrays['lever'][i]-arrays['lever'][j]))
        if distance<.003:continue
        pair=dict(i=i,j=int(j),group_A=int(arrays['group'][i]),group_B=int(arrays['group'][j]),
            row_A=int(arrays['original_row'][i]),row_B=int(arrays['original_row'][j]),
            pressure_relative_difference=float(abs(arrays['pressure_sum'][i]-arrays['pressure_sum'][j])/min(arrays['pressure_sum'][i],arrays['pressure_sum'][j])),
            normal_relative_difference=float(abs(arrays['normal'][i]-arrays['normal'][j])/min(arrays['normal'][i],arrays['normal'][j])),
            external_label_distance=distance,
            centroid_distance=float(np.linalg.norm(arrays['centroid'][i]-arrays['centroid'][j])),
            raw_L2_distance=float(np.linalg.norm(arrays['weights'][i]-arrays['weights'][j])))
        pairs.append(pair)
    pairs.sort(key=lambda p:p['external_label_distance'],reverse=True)
    # Disjoint selected frames, cap 100 pairs; retain the full candidate list.
    selected=[]
    for p in pairs:
        if p['i'] in seen or p['j'] in seen:continue
        selected.append(p);seen.update([p['i'],p['j']])
        if len(selected)==100:break
    save('candidate_pairs.json',pairs);save('selected_pairs.json',selected)
    np.savez_compressed(OUT/'screened_real_frames.npz',**arrays)
    if selected:
        ia=np.array([p['i'] for p in selected]);ib=np.array([p['j'] for p in selected])
        np.savez_compressed(OUT/'load_matched_real_pairs.npz',
            pressure_A=arrays['weights'][ia],pressure_B=arrays['weights'][ib],
            wrench_A=arrays['ft'][ia],wrench_B=arrays['ft'][ib],
            source_group_A=arrays['group'][ia],source_group_B=arrays['group'][ib],
            original_row_A=arrays['original_row'][ia],original_row_B=arrays['original_row'][ib])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(8,10))
        for row,p in enumerate(selected[:3]):
            maximum=max(arrays['weights'][p['i']].max(),arrays['weights'][p['j']].max())
            for col,index in enumerate([p['i'],p['j']]):
                ax=axes[row,col]
                ax.imshow(arrays['weights'][index],vmin=0,vmax=maximum,cmap='viridis')
                ax.set_title(f"obj{arrays['group'][index]}, row {arrays['original_row'][index]}\nP={arrays['pressure_sum'][index]:.1f}; |Fz|={arrays['normal'][index]:.2f}\nATI ratio=({arrays['lever'][index,0]:.4f}, {arrays['lever'][index,1]:.4f})")
                ax.set_xlabel('Taxel column');ax.set_ylabel('Taxel row')
        fig.suptitle('Real pressure maps matched within 5% total response and normal load',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'matched_real_maps.png',dpi=160);plt.close(fig)
    summary=dict(source_sequences_inspected=6,total_raw_rows=sum(m['rows'] for _,m in records),
        calibration_qualified_sequences=sum(a is not None for a,_ in records),
        sampled_contact_rows=len(pos),eligible_stable_windows=sum(m.get('stable_window_count',0) for _,m in records),
        eligible_low_shear_stable_windows=sum(m.get('stable_windows_low_shear_count',0) for _,m in records),
        candidate_pair_count=len(pairs),selected_disjoint_frame_pair_count=len(selected),
        selected_pairs_centroid_distance_quantiles=np.quantile([p['centroid_distance'] for p in selected],[0,.5,1]).tolist() if selected else None,
        conclusion='Real data candidate; no proof of tipping labels, causal necessity, full11D sufficiency or closed-loop benefit')
    save('summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
