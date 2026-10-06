"""Reproducible mathematical checks and a LIMITED real-data scalar screening.
Run: python outputs/validate_contact_operator.py
Requires numpy. Downloads 12 small CC-BY-NC-4.0 HTT episodes into work/htt.
Does NOT assign imaginary spatial coordinates to the 72 HTT channels.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
DATA = ROOT / 'work' / 'htt'
DATA.mkdir(parents=True, exist_ok=True)
BASE = 'https://huggingface.co/datasets/AllenBi21/HTT-dataset'
REVISION = 'b3d6f0a275dfc949cbe6c100e4ca767e4ad8bf59'


def descriptor(w, xy, threshold=0.0, pref=1.0):
    """8 static coordinates; 3 causal deltas supplied separately.
    Defines zero output on zero mass as a convention, not a measured centroid.
    Moment denominator uses exact positive mass; orientation uses repo epsilon.
    """
    w = np.asarray(w, dtype=float)
    assert np.isfinite(w).all() and (w >= 0).all()
    p = w.sum()
    if p == 0:
        return np.zeros(8)
    c = w @ xy / p
    r = xy - c
    cov = (r.T * w) @ r / p
    eig = np.linalg.eigvalsh(cov)[::-1]
    a, b, d = cov[0, 0], cov[0, 1], cov[1, 1]
    gap = np.hypot(a-d, 2*b)
    eps = 1e-12
    q = gap / (a+d+eps)
    o = q * np.array([a-d, 2*b]) / (gap+eps)
    return np.r_[np.log1p(p/pref), c, np.mean(w > threshold),
                 np.sqrt(np.maximum(eig, 0)), o]


def mathematical_checks():
    xy = np.c_[np.linspace(-1, 1, 5), np.zeros(5)]
    v = np.array([1, -4, 6, -4, 1.])
    wa, wb = 10+v, 10-v
    sa, sb = descriptor(wa, xy), descriptor(wb, xy)
    # Two constant histories also have identical three difference coordinates.
    full_a, full_b = np.r_[sa, [0, 0, 0]], np.r_[sb, [0, 0, 0]]
    assert np.allclose(full_a, full_b, atol=1e-12, rtol=0)
    square = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1.]])
    iso = descriptor(np.ones(4), square)
    assert np.linalg.norm(iso[6:8]) == 0
    rng = np.random.default_rng(20261006)
    errors = []
    for _ in range(1000):
        points = rng.uniform(-.5, .5, (24, 2))
        w = rng.uniform(.1, 2, 24)
        s = descriptor(w, points)
        scaled = descriptor(3*w, points)
        translated = descriptor(w, points+np.array([.1, -.2]))
        errors.extend([np.max(np.abs(s[1:]-scaled[1:])),
                       np.max(np.abs(translated[1:3]-s[1:3]-[.1, -.2])),
                       np.max(np.abs(translated[4:]-s[4:]))])
    assert max(errors) < 1e-10
    # Fixed threshold has a genuine jump under arbitrarily small perturbations.
    x1, x2 = np.ones(4), np.ones(4)
    x1[0], x2[0] = 1-1e-9, 1+1e-9
    area_jump = descriptor(x2, square, 1)[3]-descriptor(x1, square, 1)[3]
    assert area_jump == .25
    return dict(random_positive_maps=1000, maximum_identity_error=max(errors),
                collision_weights_A=wa.tolist(), collision_weights_B=wb.tolist(),
                collision_descriptor_max_error=float(np.max(abs(full_a-full_b))),
                collision_raw_L2_distance=float(np.linalg.norm(wa-wb)),
                isotropic_orientation=iso[6:8].tolist(),
                threshold_area_jump=float(area_jump),
                note='Synthetic algebraic checks; not sensor or control evidence.')


def get_episode(probe, index):
    path = f'force/xela/processed/p{probe}_static/sync_0_press_{index}.npz'
    dst = DATA / f'p{probe}_press_{index}.npz'
    url = BASE + '/resolve/' + REVISION + '/' + path
    if not dst.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    content = r.read()
                dst.write_bytes(content)
                break
            except OSError:
                if attempt == 2:
                    raise
    with np.load(dst, allow_pickle=False) as z:
        delta = z['tactile'].astype(float)-z['ref_tactile'].astype(float)
        target = np.abs((z['6d_force']-z['ref_force'])[:, 2])
        assert delta.shape[1] == 72 and len(delta) == len(target)
        assert np.isfinite(delta).all() and np.isfinite(target).all()
        # Mirrors official loader's absolute reference subtraction, without
        # normalization/clipping. Sum includes all 72 channels: NOT normal force.
        proxy = np.abs(delta).sum(axis=1)
    meta = dict(path=path, url=url, frames=len(target), probe=probe,
                split='test' if (probe, index) in ((1, 12), (2, 102), (3, 12), (4, 54)) else 'train',
                sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    return proxy, target, meta


def real_scalar_screen():
    tasks = [(p, i) for p, indices in [(1, (10, 11, 12)), (2, (100, 101, 102)),
                                      (3, (10, 11, 12)), (4, (52, 53, 54))]
             for i in indices]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(lambda t: get_episode(*t), tasks))
    manifest_path = DATA/'MANIFEST.sha256'
    if not manifest_path.exists():
        with urllib.request.urlopen(BASE+'/resolve/'+REVISION+'/MANIFEST.sha256', timeout=20) as r:
            manifest_path.write_bytes(r.read())
    expected = {line.split(maxsplit=1)[1].lstrip('*').removeprefix('./'):line.split()[0]
                for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()}
    matches = {e[2]['path']: expected.get(e[2]['path']) == e[2]['sha256'] for e in episodes}
    assert all(matches.values()), 'Published SHA-256 manifest mismatch'
    train = [e for e in episodes if e[2]['split'] == 'train']
    test = [e for e in episodes if e[2]['split'] == 'test']
    p_train = np.concatenate([e[0] for e in train])
    y_train = np.concatenate([e[1] for e in train])
    pref = float(np.median(p_train[p_train > 0]))
    metrics = {}
    for name, transform in [('L1_proxy', lambda p:p),
                            ('log_L1_proxy', lambda p:np.log1p(p/pref))]:
        x = transform(p_train)
        coef = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y_train, rcond=None)[0]
        per_trial = []
        for p, y, meta in test:
            pred = np.c_[transform(p), np.ones(len(p))] @ coef
            residual = y-pred
            baseline = float(np.mean(y_train))
            per_trial.append(dict(probe=meta['probe'], frames=len(y),
                pearson_r=float(np.corrcoef(transform(p), y)[0, 1]),
                mae_N=float(np.mean(abs(residual))),
                rmse_N=float(np.sqrt(np.mean(residual**2))),
                train_mean_baseline_mae_N=float(np.mean(abs(y-baseline)))))
        metrics[name] = dict(coefficients=coef.tolist(), test_trials=per_trial,
            macro_MAE_N=float(np.mean([x['mae_N'] for x in per_trial])))
    return dict(dataset=BASE, dataset_revision=REVISION,
        publisher_manifest_sha256_matches=matches, episode_count=len(episodes),
        frame_count=sum(e[2]['frames'] for e in episodes), train_episodes=8,
        test_episodes=4, P_ref_train_only=pref, metrics=metrics,
        manifests=[e[2] for e in episodes],
        limitation='Scalar screening only: all-channel L1 is not the project normal-taxel P. '
        'No verified spatial/channel map, no timestamps, no full 11D, no OOD, no control. '
        'Four held-out trials, same probes as train; frames are correlated. '
        'Dataset reference frame provenance not independently verified.')


if __name__ == '__main__':
    result = dict(repository_commit='d6871a30f27dc1144ea4b3b073e1f58206f44a00',
                  mathematical=mathematical_checks(), real_scalar=real_scalar_screen())
    (OUT/'operator_validation_results.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'real_scalar'}, indent=2))
    print(json.dumps({k:v for k,v in result['real_scalar'].items() if k != 'manifests'}, indent=2))
