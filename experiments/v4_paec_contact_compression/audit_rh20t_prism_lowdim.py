#!/usr/bin/env python3
"""
V4.2 Pre-Scaling Audit Tool for RH20T_lowdim & PRISM (and Canonical 6D Contact Schema).

Two modes:
  1. --mode remote-list
       Uses lightweight REST API calls (zero bulk download) to inspect the remote
       directory trees & file sizes of `tonyfanggalaxies/RH20T` (specifically RH20T_lowdim)
       and `Tengbo-Yu/PRISM`.
  2. --mode audit-local --data-dir <path>
       Audits a local sample (e.g., 100-500 episodes) of RH20T_lowdim, PRISM, or .npz/.h5
       contact trajectories BEFORE downloading/training at 10k-50k scale:
         - Schema / tensor shapes / dtypes
         - True sampling rate (dt_ms, Hz) & F/T vs pose frequency alignment
         - NaN/Inf, non-zero fraction, saturation check
         - SI unit sanity check (N / N*m percentiles)
         - Contact-active density (||F|| > 1.0 N or ||dF|| > 0.2 N)
         - Gravity-orientation coupling warning (low-speed wrench vs orientation change)
         - Task / robot-config diversity for Held-Out Task & Held-Out Embodiment splits
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# 1. REMOTE METADATA INSPECTION (ZERO BULK DOWNLOAD)
# ==============================================================================
def _fetch_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ICRA2028-PAEC-Audit/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0:
            return f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PB"


def inspect_hf_tree(repo_id: str, subpath: str = "", timeout: int = 20, hf_endpoint: str = "") -> Dict[str, Any]:
    host = (hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    base = f"{host}/api/datasets/{repo_id}/tree/main"
    url = f"{base}/{subpath.strip('/')}" if subpath.strip("/") else base
    entries = _fetch_json(url, timeout=timeout)
    files, dirs = [], []
    total_bytes = 0
    for e in entries:
        et = e.get("type", "")
        path = e.get("path", "")
        size = int(e.get("size", 0) or 0)
        if et == "directory":
            dirs.append(path)
        else:
            files.append({"path": path, "size": size})
            total_bytes += size
    return {
        "repo_id": repo_id,
        "subpath": subpath,
        "url": url,
        "n_dirs": len(dirs),
        "n_files": len(files),
        "total_file_bytes": total_bytes,
        "total_file_human": _fmt_bytes(total_bytes),
        "dirs_sample": dirs[:30],
        "files_sample": [{"path": f["path"], "size_human": _fmt_bytes(f["size"])} for f in files[:30]],
    }


def inspect_github_repo(owner_repo: str, subpath: str = "", timeout: int = 20) -> Dict[str, Any]:
    base = f"https://api.github.com/repos/{owner_repo}/contents"
    url = f"{base}/{subpath.strip('/')}" if subpath.strip("/") else base
    entries = _fetch_json(url, timeout=timeout)
    if isinstance(entries, dict):
        entries = [entries]
    items = [
        {"name": e.get("name"), "type": e.get("type"), "size": int(e.get("size", 0) or 0)}
        for e in entries
    ]
    return {"repo": owner_repo, "subpath": subpath, "url": url, "items": items[:40]}


def run_remote_list(hf_repo: str, hf_subpath: str, github_repo: str, timeout: int, hf_endpoint: str = "") -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "remote-list", "hf_root": None, "hf_subpath": None, "github": None}
    print(f"\n=== [Remote Audit 1/3] HuggingFace Root: {hf_repo} ===")
    try:
        r_root = inspect_hf_tree(hf_repo, "", timeout=timeout, hf_endpoint=hf_endpoint)
        report["hf_root"] = r_root
        print(f"  Dirs ({r_root['n_dirs']}): {r_root['dirs_sample']}")
        print(f"  Top files ({r_root['n_files']}, {r_root['total_file_human']}): {r_root['files_sample'][:10]}")
    except Exception as e:
        report["hf_root"] = {"error": str(e)}
        print(f"  [WARN] Could not query HF root ({hf_repo}): {e}")

    if hf_subpath:
        print(f"\n=== [Remote Audit 2/3] HuggingFace Subpath: {hf_repo}/{hf_subpath} ===")
        try:
            r_sub = inspect_hf_tree(hf_repo, hf_subpath, timeout=timeout, hf_endpoint=hf_endpoint)
            report["hf_subpath"] = r_sub
            print(f"  Dirs ({r_sub['n_dirs']}): {r_sub['dirs_sample'][:20]}")
            print(f"  Files ({r_sub['n_files']}, {r_sub['total_file_human']}): {r_sub['files_sample'][:15]}")
        except Exception as e:
            report["hf_subpath"] = {"error": str(e)}
            print(f"  [WARN] Could not query HF subpath ({hf_repo}/{hf_subpath}): {e}")

    if github_repo:
        print(f"\n=== [Remote Audit 3/3] GitHub Repo: {github_repo} ===")
        try:
            r_gh = inspect_github_repo(github_repo, "", timeout=timeout)
            report["github"] = r_gh
            for it in r_gh["items"][:20]:
                print(f"  [{it['type']:4s}] {it['name']} ({_fmt_bytes(it['size'])})")
        except Exception as e:
            report["github"] = {"error": str(e)}
            print(f"  [WARN] Could not query GitHub ({github_repo}): {e}")
    return report


# ==============================================================================
# 2. LOCAL PHYSICAL & SCHEMA AUDIT (RH20T / PRISM / CANONICAL NPZ / H5)
# ==============================================================================
WRENCH_KEY_HINTS = ("wrench_raw", "tcp_wrench", "force_torque", "wrench", "ft_sensor", "force")
POSE_KEY_HINTS = ("tcp_pose", "tcp_base", "ee_pose", "eef_pose", "proprio_state", "pose", "cartesian")
TS_KEY_HINTS = ("timestamp", "timestamps", "time_sec", "time_s", "time")


def _find_matching_key(keys: List[str], hints: Tuple[str, ...]) -> Optional[str]:
    low = {k.lower(): k for k in keys}
    for h in hints:
        if h in low:
            return low[h]
    for h in hints:
        pat = re.compile(rf"(^|[/_-]){re.escape(h)}($|[/_-])")
        for kl, orig in low.items():
            if "img" in kl or "rgb" in kl or "depth" in kl:
                continue
            if pat.search(kl):
                return orig
    return None


def _extract_cfg_and_task(path: Path) -> Tuple[str, str]:
    s = str(path).replace("\\", "/")
    cfg_m = re.search(r"(cfg[_-]?\d+|robot[_-]?\w+)", s, flags=re.IGNORECASE)
    task_m = re.search(r"(task[_-]?\d+)", s, flags=re.IGNORECASE)
    cfg = cfg_m.group(1).lower() if cfg_m else "unknown_cfg"
    task = task_m.group(1).lower() if task_m else path.parent.name
    return cfg, task


def _audit_single_episode_arrays(
    ep_id: str,
    cfg: str,
    task: str,
    wrench: Optional[np.ndarray],
    pose: Optional[np.ndarray],
    ts_ft: Optional[np.ndarray],
    ts_pose: Optional[np.ndarray],
    force_active_thresh_n: float = 1.0,
    df_active_thresh_n: float = 0.2,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "ep_id": ep_id,
        "cfg": cfg,
        "task": task,
        "has_wrench": wrench is not None,
        "has_pose": pose is not None,
        "wrench_shape": list(wrench.shape) if wrench is not None else None,
        "pose_shape": list(pose.shape) if pose is not None else None,
    }
    if wrench is not None and wrench.ndim == 2 and wrench.shape[0] >= 5 and wrench.shape[1] >= 6:
        w = wrench[:, :6].astype(np.float64)
        finite_mask = np.isfinite(w).all(axis=-1)
        rec["n_frames_ft"] = int(w.shape[0])
        rec["nonfinite_frac"] = round(float(1.0 - finite_mask.mean()), 6)
        w = w[finite_mask]
        if len(w) >= 5:
            f_norm = np.linalg.norm(w[:, :3], axis=-1)
            t_norm = np.linalg.norm(w[:, 3:6], axis=-1)
            df_norm = np.linalg.norm(np.diff(w[:, :3], axis=0, prepend=w[:1, :3]), axis=-1)
            rec["nonzero_frac"] = round(float((f_norm > 1e-4).mean()), 4)
            rec["contact_active_frac"] = round(
                float(((f_norm > force_active_thresh_n) | (df_norm > df_active_thresh_n)).mean()), 4
            )
            rec["force_p50_n"] = round(float(np.percentile(f_norm, 50)), 4)
            rec["force_p95_n"] = round(float(np.percentile(f_norm, 95)), 4)
            rec["force_max_n"] = round(float(f_norm.max()), 4)
            rec["torque_p95_nm"] = round(float(np.percentile(t_norm, 95)), 4)
            # Saturation heuristic: fraction of consecutive identical nonzero 6D readings
            same_next = (np.linalg.norm(np.diff(w, axis=0), axis=-1) < 1e-7) & (f_norm[1:] > 1e-4)
            rec["stuck_repeat_frac"] = round(float(same_next.mean()), 4)

            # Gravity-orientation coupling diagnostic (only if pose & wrench lengths match and pose has orientation)
            if pose is not None and pose.ndim == 2 and len(pose) == len(w) and pose.shape[1] >= 6:
                dp = np.linalg.norm(np.diff(pose[:, :3], axis=0, prepend=pose[:1, :3]), axis=-1)
                dr = np.linalg.norm(np.diff(pose[:, 3:], axis=0, prepend=pose[:1, 3:]), axis=-1)
                low_trans = dp < np.percentile(dp, 40)
                if low_trans.sum() >= 20 and float(dr[low_trans].std()) > 1e-6 and float(df_norm[low_trans].std()) > 1e-6:
                    corr = float(np.corrcoef(dr[low_trans], df_norm[low_trans])[0, 1])
                    rec["low_speed_rot_force_corr"] = round(corr, 4)

    for label, ts in (("ft", ts_ft), ("pose", ts_pose)):
        if ts is not None and ts.ndim == 1 and len(ts) >= 5:
            dt = np.diff(ts.astype(np.float64))
            # Convert ms/us/ns timestamps heuristically to seconds
            med_raw = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
            scale = 1.0
            if med_raw > 1e6:
                scale = 1e-9
            elif med_raw > 1e3:
                scale = 1e-6
            elif med_raw > 1.0:
                scale = 1e-3
            dt_s = dt * scale
            valid = dt_s[(dt_s > 1e-4) & (dt_s < 2.0)]
            if len(valid) > 0:
                med_dt = float(np.median(valid))
                rec[f"{label}_dt_ms"] = round(med_dt * 1000.0, 2)
                rec[f"{label}_hz"] = round(1.0 / med_dt, 2)
                rec[f"{label}_jitter_ms"] = round(float(np.percentile(np.abs(valid - med_dt), 95) * 1000.0), 2)
    return rec


def _load_npz_episodes(npz_path: Path, max_eps: int) -> List[Dict[str, Any]]:
    raw = np.load(npz_path, allow_pickle=False)
    keys = list(raw.keys())
    wk = _find_matching_key(keys, WRENCH_KEY_HINTS)
    pk = _find_matching_key(keys, POSE_KEY_HINTS)
    tk = _find_matching_key(keys, TS_KEY_HINTS)

    w_all = raw[wk] if wk else None
    p_all = raw[pk] if pk else None
    t_all = raw[tk] if tk else None
    cfg, task = _extract_cfg_and_task(npz_path)

    records = []
    if "ep_starts" in keys and "ep_ends" in keys:
        starts, ends = raw["ep_starts"], raw["ep_ends"]
        for idx, (s, e) in enumerate(zip(starts[:max_eps], ends[:max_eps])):
            s, e = int(s), int(e)
            records.append(
                _audit_single_episode_arrays(
                    ep_id=f"{npz_path.stem}:ep{idx}",
                    cfg=cfg,
                    task=task,
                    wrench=w_all[s:e] if w_all is not None else None,
                    pose=p_all[s:e] if p_all is not None else None,
                    ts_ft=t_all[s:e] if t_all is not None else None,
                    ts_pose=t_all[s:e] if t_all is not None else None,
                )
            )
    else:
        records.append(
            _audit_single_episode_arrays(
                ep_id=npz_path.stem,
                cfg=cfg,
                task=task,
                wrench=w_all,
                pose=p_all,
                ts_ft=t_all,
                ts_pose=t_all,
            )
        )
    return records


def _load_rh20t_dir_episode(ep_dir: Path) -> Optional[Dict[str, Any]]:
    """Inspect an RH20T-style episode folder containing .npy files (e.g. force_torque.npy, tcp_base.npy)."""
    npy_files = {p.stem.lower(): p for p in ep_dir.glob("*.npy")}
    if not npy_files:
        return None
    keys = list(npy_files.keys())
    wk = _find_matching_key(keys, WRENCH_KEY_HINTS)
    pk = _find_matching_key(keys, POSE_KEY_HINTS)
    if wk is None and pk is None:
        return None

    wrench = np.load(npy_files[wk], allow_pickle=True) if wk else None
    pose = np.load(npy_files[pk], allow_pickle=True) if pk else None
    # Handle RH20T dict-saved .npy objects gracefully
    ts_ft, ts_pose = None, None
    if isinstance(wrench, np.ndarray) and wrench.ndim == 0 and isinstance(wrench.item(), dict):
        d = wrench.item()
        ts_ft = np.asarray(sorted(d.keys()), dtype=np.float64)
        wrench = np.asarray([d[k] for k in sorted(d.keys())], dtype=np.float32)
    if isinstance(pose, np.ndarray) and pose.ndim == 0 and isinstance(pose.item(), dict):
        d = pose.item()
        ts_pose = np.asarray(sorted(d.keys()), dtype=np.float64)
        pose = np.asarray([d[k] for k in sorted(d.keys())], dtype=np.float32)

    cfg, task = _extract_cfg_and_task(ep_dir)
    return _audit_single_episode_arrays(
        ep_id=ep_dir.name,
        cfg=cfg,
        task=task,
        wrench=wrench if isinstance(wrench, np.ndarray) else None,
        pose=pose if isinstance(pose, np.ndarray) else None,
        ts_ft=ts_ft,
        ts_pose=ts_pose,
    )


def _load_h5_episodes(h5_path: Path, max_eps: int) -> List[Dict[str, Any]]:
    try:
        import h5py
    except ImportError:
        print(f"  [WARN] h5py not installed; skipping {h5_path}")
        return []
    records = []
    cfg, task = _extract_cfg_and_task(h5_path)
    with h5py.File(str(h5_path), "r") as f:
        flat_keys = []

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                flat_keys.append(name)

        f.visititems(_visit)
        wk = _find_matching_key(flat_keys, WRENCH_KEY_HINTS)
        pk = _find_matching_key(flat_keys, POSE_KEY_HINTS)
        tk = _find_matching_key(flat_keys, TS_KEY_HINTS)
        records.append(
            _audit_single_episode_arrays(
                ep_id=h5_path.stem,
                cfg=cfg,
                task=task,
                wrench=np.asarray(f[wk]) if wk else None,
                pose=np.asarray(f[pk]) if pk else None,
                ts_ft=np.asarray(f[tk]) if tk else None,
                ts_pose=np.asarray(f[tk]) if tk else None,
            )
        )
    return records[:max_eps]


def aggregate_audit_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_episodes_audited": 0, "go_no_go_flags": ["NO_EPISODES_FOUND"]}

    def _vals(key: str) -> List[float]:
        return [float(r[key]) for r in records if key in r and r[key] is not None]

    cfg_counts = Counter(r["cfg"] for r in records)
    task_counts = Counter(r["task"] for r in records)
    ft_hz = _vals("ft_hz")
    pose_hz = _vals("pose_hz")
    active = _vals("contact_active_frac")
    nonzero = _vals("nonzero_frac")
    f_p95 = _vals("force_p95_n")
    f_max = _vals("force_max_n")
    t_p95 = _vals("torque_p95_nm")
    rot_corr = _vals("low_speed_rot_force_corr")
    nonfin = _vals("nonfinite_frac")
    stuck = _vals("stuck_repeat_frac")

    flags = []
    if len(_vals("n_frames_ft")) == 0:
        flags.append("CRITICAL: NO_6D_WRENCH_FOUND")
    if nonfin and max(nonfin) > 0.0:
        flags.append(f"WARN: NONFINITE_VALUES_MAX={max(nonfin):.4f}")
    if nonzero and float(np.median(nonzero)) < 0.5:
        flags.append(f"WARN: LOW_NONZERO_WRENCH_MEDIAN={np.median(nonzero):.3f}")
    if active and float(np.median(active)) < 0.2:
        flags.append(f"WARN: SPARSE_CONTACT_MEDIAN={np.median(active):.3f} (stratified contact filtering required)")
    if f_max and float(np.percentile(f_max, 95)) > 300.0:
        flags.append(f"WARN: FORCE_SCALE_SUSPECT_P95_MAX={np.percentile(f_max, 95):.1f} (check SI Newtons vs raw ADC)")
    if ft_hz and pose_hz and abs(float(np.median(ft_hz)) - float(np.median(pose_hz))) / max(float(np.median(pose_hz)), 1.0) > 0.2:
        flags.append(f"WARN: FREQ_MISMATCH_FT={np.median(ft_hz):.1f}Hz_vs_POSE={np.median(pose_hz):.1f}Hz (resample before PAEC)")
    if rot_corr and float(np.median(np.abs(rot_corr))) > 0.45:
        flags.append(f"WARN: GRAVITY_ORIENTATION_COUPLING_MEDIAN_ABS_R={np.median(np.abs(rot_corr)):.3f} (check tool gravity comp)")
    if not flags:
        flags.append("PASS: ALL_CORE_PHYSICAL_CHECKS_CLEAN")

    return {
        "n_episodes_audited": n,
        "n_with_6d_wrench": sum(1 for r in records if r.get("n_frames_ft", 0) > 0),
        "n_with_pose": sum(1 for r in records if r.get("has_pose")),
        "n_unique_configs": len(cfg_counts),
        "n_unique_tasks": len(task_counts),
        "top_configs": dict(cfg_counts.most_common(10)),
        "top_tasks": dict(task_counts.most_common(10)),
        "ft_hz_median": round(float(np.median(ft_hz)), 2) if ft_hz else None,
        "pose_hz_median": round(float(np.median(pose_hz)), 2) if pose_hz else None,
        "nonzero_wrench_median": round(float(np.median(nonzero)), 4) if nonzero else None,
        "contact_active_median": round(float(np.median(active)), 4) if active else None,
        "contact_active_p25": round(float(np.percentile(active, 25)), 4) if active else None,
        "force_p95_median_n": round(float(np.median(f_p95)), 4) if f_p95 else None,
        "torque_p95_median_nm": round(float(np.median(t_p95)), 4) if t_p95 else None,
        "stuck_repeat_median": round(float(np.median(stuck)), 4) if stuck else None,
        "low_speed_rot_force_abs_corr_median": round(float(np.median(np.abs(rot_corr))), 4) if rot_corr else None,
        "go_no_go_flags": flags,
    }


def run_local_audit(data_dir: Path, max_episodes: int) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    if data_dir.is_file():
        if data_dir.suffix == ".npz":
            records.extend(_load_npz_episodes(data_dir, max_episodes))
        elif data_dir.suffix in (".h5", ".hdf5"):
            records.extend(_load_h5_episodes(data_dir, max_episodes))
    else:
        for p in sorted(data_dir.rglob("*")):
            if len(records) >= max_episodes:
                break
            if p.is_file() and p.suffix == ".npz":
                records.extend(_load_npz_episodes(p, max_episodes - len(records)))
            elif p.is_file() and p.suffix in (".h5", ".hdf5"):
                records.extend(_load_h5_episodes(p, max_episodes - len(records)))
            elif p.is_dir():
                rec = _load_rh20t_dir_episode(p)
                if rec is not None:
                    records.append(rec)

    summary = aggregate_audit_records(records[:max_episodes])
    print("\n=== Local Physical Contact Audit Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return {"mode": "audit-local", "data_dir": str(data_dir), "summary": summary, "sample_records": records[:20]}


def main():
    parser = argparse.ArgumentParser(description="V4.2 RH20T_lowdim & PRISM Schema/Physics Audit Tool")
    parser.add_argument("--mode", type=str, required=True, choices=["remote-list", "audit-local"])
    parser.add_argument("--hf-repo", type=str, default="tonyfanggalaxies/RH20T")
    parser.add_argument("--hf-subpath", type=str, default="RH20T_lowdim")
    parser.add_argument("--hf-endpoint", type=str, default="", help="Optional HF mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--github-repo", type=str, default="Tengbo-Yu/PRISM")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="", help="Local directory or .npz/.h5 file for --mode audit-local")
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if args.mode == "remote-list":
        out = run_remote_list(args.hf_repo, args.hf_subpath, args.github_repo, args.timeout, hf_endpoint=args.hf_endpoint)
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required for --mode audit-local")
        out = run_local_audit(Path(args.data_dir), args.max_episodes)

    if args.out_json:
        op = Path(args.out_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved audit JSON to {op}")


if __name__ == "__main__":
    main()
