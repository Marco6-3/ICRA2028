#!/usr/bin/env python3
"""
Stream-extract Canonical 6D Contact Trajectories from RH20T_lowdim tar.gz archives.

Pipes `curl.exe -sL <url>` directly into `tarfile.open(..., mode="r|gz")`,
skipping all .wav audio files and unneeded arrays in RAM so zero raw tarballs hit disk.

Converts each episode to Canonical 6D Contact Schema (20 Hz uniform grid, TCP local frame, SI units):
  - contact_obs (N, 6): [Fx, Fy, Fz, Tx, Ty, Tz] in TCP local frame (N, N*m)
  - proprio_state (N, 10): [pos(3), rot6d(6), grip_m(1)]
  - proprio_delta / action_next (N, 7): [dp_tcp(3 m), dr_tcp(3 rad), dg(1 m)] at 20 Hz
  - per-episode metadata: ep_cfg, ep_task, ep_scene, ep_user, ep_calib_quality, ep_contact_active_frac, ep_force_p95_n
"""

import argparse
import io
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CURL_BIN = "curl.exe" if os.name == "nt" else "curl"


NEEDED_FILES = {
    "metadata.json",
    "force_torque_base.npy",
    "tcp_base.npy",
    "gripper.npy",
}


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert (T, 3, 3) rotation matrices to (T, 6) 6D rotation representation."""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1).astype(np.float32)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    r = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    sin_t = np.sin(theta)[..., None]
    small = theta[..., None] < 1e-4
    scale = np.where(small, 0.5, theta[..., None] / np.where(small, 1.0, 2.0 * sin_t))
    return r * scale


def parse_ep_dirname(ep_name: str) -> Dict[str, str]:
    m = re.match(r"(task_\d+)_(user_\d+)_(scene_\d+)_(cfg_\d+)", ep_name)
    if m:
        return {"task": m.group(1), "user": m.group(2), "scene": m.group(3), "cfg": m.group(4)}
    return {"task": ep_name, "user": "unknown", "scene": "unknown", "cfg": "unknown"}


def process_episode_blobs(
    ep_name: str,
    blobs: Dict[str, bytes],
    target_hz: float = 20.0,
    max_force_n: float = 250.0,
    min_duration_s: float = 1.5,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if "force_torque_base.npy" not in blobs or "tcp_base.npy" not in blobs:
        return None, "missing_ft_or_tcp"

    meta_json = {}
    if "metadata.json" in blobs:
        try:
            meta_json = json.loads(blobs["metadata.json"].decode("utf-8", errors="ignore"))
        except Exception:
            meta_json = {}

    try:
        ftb = np.load(io.BytesIO(blobs["force_torque_base.npy"]), allow_pickle=True).item()
        tcpb = np.load(io.BytesIO(blobs["tcp_base.npy"]), allow_pickle=True).item()
        grp = np.load(io.BytesIO(blobs["gripper.npy"]), allow_pickle=True).item() if "gripper.npy" in blobs else {}
    except Exception:
        return None, "npy_decode_err"

    common_cams = [c for c in ftb.keys() if c in tcpb and len(ftb[c]) >= 20 and len(tcpb[c]) >= 20]
    if not common_cams:
        return None, "short_or_no_cam"
    cam = max(common_cams, key=lambda c: min(len(ftb[c]), len(tcpb[c])))

    ft_list = ftb[cam]
    tcp_list = tcpb[cam]
    ft_map = {}
    for x in ft_list:
        if isinstance(x, dict) and "timestamp" in x and "zeroed" in x:
            arr = np.asarray(x["zeroed"], dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 6:
                ft_map[int(x["timestamp"])] = arr[:6]
    tcp_map = {}
    for x in tcp_list:
        if isinstance(x, dict) and "timestamp" in x and "tcp" in x:
            arr = np.asarray(x["tcp"], dtype=np.float64).reshape(-1)
            if arr.shape[0] >= 7:
                tcp_map[int(x["timestamp"])] = arr[:7]
    ts_common = sorted(set(ft_map.keys()) & set(tcp_map.keys()))
    if len(ts_common) < 20:
        return None, "short_synced"

    ts_ms = np.asarray(ts_common, dtype=np.float64)
    ts_s = (ts_ms - ts_ms[0]) * 1e-3
    dt_raw = np.diff(ts_s)
    if np.any(dt_raw <= 0):
        return None, "nonmono_ts"
    med_dt = float(np.median(dt_raw))
    max_dt = float(np.max(dt_raw))
    if med_dt > 0.25 or max_dt > 0.75 or ts_s[-1] < min_duration_s:
        return None, "bad_dt_or_short"

    try:
        w_base = np.stack([ft_map[t] for t in ts_common], axis=0)
        tcp = np.stack([tcp_map[t] for t in ts_common], axis=0)
    except Exception:
        return None, "bad_shape"
    if w_base.shape[1] < 6 or tcp.shape[1] < 7:
        return None, "bad_shape"
    w_base = w_base[:, :6]
    if not (np.isfinite(w_base).all() and np.isfinite(tcp).all()):
        return None, "nonfinite"

    f_norm_raw = np.linalg.norm(w_base[:, :3], axis=-1)
    if float(f_norm_raw.max()) > max_force_n or float(np.linalg.norm(w_base[:, 3:6], axis=-1).max()) > 50.0:
        return None, "force_outlier"

    # RH20T_api uses transforms3d.quaternions.quat2mat([w, x, y, z]) -> wxyz!
    q_wxyz = tcp[:, 3:7].copy()
    q_norm = np.linalg.norm(q_wxyz, axis=-1, keepdims=True)
    if np.any(q_norm < 1e-6):
        return None, "zero_quat"
    q_wxyz = q_wxyz / q_norm
    # Enforce hemisphere continuity before SLERP
    for i in range(1, len(q_wxyz)):
        if float(np.dot(q_wxyz[i], q_wxyz[i - 1])) < 0.0:
            q_wxyz[i:] *= -1.0
    q_xyzw = np.concatenate([q_wxyz[:, 1:4], q_wxyz[:, 0:1]], axis=-1)

    # Extract gripper width in meters (if available)
    grip_raw = np.zeros(len(ts_common), dtype=np.float64)
    if cam in grp and isinstance(grp[cam], dict) and len(grp[cam]) > 0:
        g_ts = sorted(grp[cam].keys())
        g_vals = []
        for gt in g_ts:
            info = grp[cam][gt].get("gripper_info", [0.0])
            g_vals.append(float(info[0]) if len(info) > 0 else 0.0)
        g_ts_s = (np.asarray(g_ts, dtype=np.float64) - ts_ms[0]) * 1e-3
        grip_raw = np.interp(ts_s, g_ts_s, np.asarray(g_vals, dtype=np.float64))
        # Convert mm -> m for non-Franka grippers (WSG-50 / Robotiq / Dahuan report 0..110 mm)
        if float(np.max(np.abs(grip_raw))) > 1.5:
            grip_raw = grip_raw * 1e-3

    # Rotate wrench from base frame into TCP local frame at raw timestamps, then resample to 20 Hz
    R_raw = Rotation.from_quat(q_xyzw)
    R_mat_raw = R_raw.as_matrix()  # (T_raw, 3, 3): base_from_tcp
    F_tcp_raw = np.einsum("tji,tj->ti", R_mat_raw, w_base[:, :3])
    T_tcp_raw = np.einsum("tji,tj->ti", R_mat_raw, w_base[:, 3:6])
    w_tcp_raw = np.concatenate([F_tcp_raw, T_tcp_raw], axis=-1)

    # Resample onto uniform 20 Hz grid
    dt_target = 1.0 / target_hz
    t_grid = np.arange(0.0, ts_s[-1] + 1e-9, dt_target)
    if len(t_grid) < 25:
        return None, "short_resampled"
    t_grid[-1] = min(t_grid[-1], ts_s[-1])

    pos_g = np.stack([np.interp(t_grid, ts_s, tcp[:, i]) for i in range(3)], axis=-1).astype(np.float32)
    grip_g = np.interp(t_grid, ts_s, grip_raw)[:, None].astype(np.float32)
    w_tcp_g = np.stack([np.interp(t_grid, ts_s, w_tcp_raw[:, i]) for i in range(6)], axis=-1).astype(np.float32)
    slerp = Slerp(ts_s, R_raw)
    R_g = slerp(t_grid).as_matrix().astype(np.float32)
    rot6d_g = rotmat_to_rot6d(R_g)

    proprio_state = np.concatenate([pos_g, rot6d_g, grip_g], axis=-1).astype(np.float32)

    # Local TCP frame motion increments (exact same contract as inspect_rdp_zarr.py)
    R_prev_T = np.swapaxes(R_g[:-1], -1, -2)
    dp_loc = np.einsum("lij,lj->li", R_prev_T, pos_g[1:] - pos_g[:-1])
    dR_loc = np.matmul(R_prev_T, R_g[1:])
    dr_loc = rotmat_to_axis_angle(dR_loc)
    dg = grip_g[1:] - grip_g[:-1]
    step_d = np.concatenate([dp_loc, dr_loc, dg], axis=-1).astype(np.float32)

    proprio_delta = np.zeros((len(t_grid), 7), dtype=np.float32)
    action_next = np.zeros((len(t_grid), 7), dtype=np.float32)
    proprio_delta[1:] = step_d
    action_next[:-1] = step_d

    fn = np.linalg.norm(w_tcp_g[:, :3], axis=-1)
    dfn = np.linalg.norm(np.diff(w_tcp_g[:, :3], axis=0, prepend=w_tcp_g[:1, :3]), axis=-1)
    contact_active_frac = float(((fn > 1.0) | (dfn > 0.2)).mean())
    ids = parse_ep_dirname(ep_name)

    return {
        "ep_name": ep_name,
        "cfg": ids["cfg"],
        "task": ids["task"],
        "user": ids["user"],
        "scene": ids["scene"],
        "calib_quality": int(meta_json.get("calib_quality", -99)),
        "rating": int(meta_json.get("rating", -99)),
        "orig_hz": round(1.0 / med_dt, 2),
        "contact_active_frac": round(contact_active_frac, 4),
        "force_p95_n": round(float(np.percentile(fn, 95)), 4),
        "contact_obs": w_tcp_g,
        "wrench_raw": w_tcp_g,
        "proprio_state": proprio_state,
        "proprio_delta": proprio_delta,
        "action_next": action_next,
    }, "ok"


def save_canonical_shard(out_path: Path, eps: List[Dict[str, Any]], stats_counter: Dict[str, int], source_url: str, elapsed_s: float):
    if len(eps) == 0:
        print(f"[WARN] No valid episodes extracted for {out_path}; skip write.")
        return
    contact_obs = np.concatenate([e["contact_obs"] for e in eps], axis=0)
    wrench_raw = np.concatenate([e["wrench_raw"] for e in eps], axis=0)
    proprio_state = np.concatenate([e["proprio_state"] for e in eps], axis=0)
    proprio_delta = np.concatenate([e["proprio_delta"] for e in eps], axis=0)
    action_next = np.concatenate([e["action_next"] for e in eps], axis=0)

    lengths = [len(e["contact_obs"]) for e in eps]
    ep_ends = np.cumsum(lengths, dtype=np.int64)
    ep_starts = np.concatenate([[0], ep_ends[:-1]]).astype(np.int64)

    n_ep = len(eps)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(round(0.15 * n_ep)))
    n_test = max(1, int(round(0.15 * n_ep)))
    n_train = max(1, n_ep - n_val - n_test)
    train_eps = np.sort(perm[:n_train])
    val_eps = np.sort(perm[n_train : n_train + n_val])
    test_eps = np.sort(perm[n_train + n_val :])

    train_mask = np.zeros(len(contact_obs), dtype=bool)
    for eid in train_eps:
        train_mask[int(ep_starts[eid]) : int(ep_ends[eid])] = True

    def _stats(x):
        m = x[train_mask].mean(axis=0).astype(np.float32)
        s = np.maximum(x[train_mask].std(axis=0), 1e-4).astype(np.float32)
        return m, s

    con_mean, con_std = _stats(contact_obs)
    prop_mean, prop_std = _stats(proprio_state)
    act_mean, act_std = _stats(action_next)

    meta = {
        "source": source_url,
        "n_episodes": int(n_ep),
        "total_frames": int(len(contact_obs)),
        "dt_ms": 50.0,
        "fps": 20.0,
        "wrench_dim": 6,
        "grip_force_dim": 0,
        "tactile_dim": 0,
        "tactile_keys": [],
        "contact_dim": 6,
        "proprio_dim": 10,
        "act_dim": 7,
        "n_unique_tasks": len(set(e["task"] for e in eps)),
        "n_unique_cfgs": len(set(e["cfg"] for e in eps)),
        "extract_stats": stats_counter,
        "elapsed_sec": round(elapsed_s, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        contact_obs=contact_obs,
        wrench_raw=wrench_raw,
        proprio_state=proprio_state,
        proprio_delta=proprio_delta,
        action_next=action_next,
        ep_starts=ep_starts,
        ep_ends=ep_ends,
        train_eps=train_eps,
        val_eps=val_eps,
        test_eps=test_eps,
        con_mean=con_mean,
        con_std=con_std,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
        ep_names=np.array([e["ep_name"] for e in eps]),
        ep_cfg=np.array([e["cfg"] for e in eps]),
        ep_task=np.array([e["task"] for e in eps]),
        ep_scene=np.array([e["scene"] for e in eps]),
        ep_user=np.array([e["user"] for e in eps]),
        ep_calib_quality=np.array([e["calib_quality"] for e in eps], dtype=np.int32),
        ep_rating=np.array([e["rating"] for e in eps], dtype=np.int32),
        ep_orig_hz=np.array([e["orig_hz"] for e in eps], dtype=np.float32),
        ep_contact_active_frac=np.array([e["contact_active_frac"] for e in eps], dtype=np.float32),
        ep_force_p95_n=np.array([e["force_p95_n"] for e in eps], dtype=np.float32),
        meta_json=np.array(json.dumps(meta)),
    )
    tmp_path.replace(out_path)
    print(f"[SAVED] {out_path.name}: eps={n_ep}, frames={len(contact_obs)}, tasks={meta['n_unique_tasks']}, elapsed={elapsed_s:.1f}s | {stats_counter}", flush=True)


def _get_remote_size(url: str) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [CURL_BIN, "-sIL", "--max-time", "20", url],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore")
        sizes = re.findall(r"content-length:\s*(\d+)", out, flags=re.IGNORECASE)
        if sizes:
            return int(sizes[-1])
    except Exception:
        pass
    return None


class ChunkedHTTPRangeStream(io.RawIOBase):
    """
    Fault-tolerant sequential stream backed by chunked curl HTTP Range requests.
    Avoids proxy/CDN long-lived stream disconnects and retries seamlessly.
    """

    def __init__(self, url: str, chunk_bytes: int = 16 * 1024 * 1024):
        self.url = url
        self.chunk_bytes = chunk_bytes
        self.total_size = _get_remote_size(url)
        self.offset = 0
        self.buf = b""
        self.buf_pos = 0
        self.eof = False

    def readable(self) -> bool:
        return True

    def _fetch_next_chunk(self) -> bool:
        if self.eof:
            return False
        if self.total_size is not None and self.offset >= self.total_size:
            self.eof = True
            return False
        end = self.offset + self.chunk_bytes - 1
        if self.total_size is not None:
            end = min(end, self.total_size - 1)
        for attempt in range(6):
            try:
                p = subprocess.Popen(
                    [CURL_BIN, "-sL", "--max-time", "90", "--range", f"{self.offset}-{end}", self.url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                data, err = p.communicate(timeout=70)
                if p.returncode == 0 and len(data) > 0:
                    self.buf = data
                    self.buf_pos = 0
                    self.offset += len(data)
                    if len(data) < (end - (self.offset - len(data)) + 1) and self.total_size is None:
                        self.eof = True
                    return True
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        self.eof = True
        return False

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        chunks = []
        rem = n if n > 0 else (1024 * 1024)
        while rem > 0:
            if self.buf_pos >= len(self.buf):
                if not self._fetch_next_chunk():
                    break
            avail = len(self.buf) - self.buf_pos
            take = min(rem, avail)
            chunks.append(self.buf[self.buf_pos : self.buf_pos + take])
            self.buf_pos += take
            rem -= take
            if n < 0 and self.buf_pos >= len(self.buf):
                break
        return b"".join(chunks)


def stream_cfg(
    cfg_num: int,
    out_path: Path,
    max_episodes: int,
    max_seconds: int,
    target_hz: float,
    chunk_mb: int = 16,
    save_every: int = 200,
):
    url = f"https://huggingface.co/datasets/tonyfanggalaxies/RH20T/resolve/main/RH20T_lowdim/RH20T_cfg{cfg_num}.tar.gz"
    t0 = time.time()
    rstream = ChunkedHTTPRangeStream(url, chunk_bytes=chunk_mb * 1024 * 1024)
    print(f"  [cfg{cfg_num}] remote_size={rstream.total_size} bytes | chunk={chunk_mb}MB | save_every={save_every}", flush=True)
    eps: List[Dict[str, Any]] = []
    stats_counter: Dict[str, int] = {}
    cur_ep: Optional[str] = None
    cur_blobs: Dict[str, bytes] = {}
    seen_eps = 0

    def _flush_cur():
        nonlocal seen_eps
        if cur_ep is None:
            return
        seen_eps += 1
        rec, reason = process_episode_blobs(cur_ep, cur_blobs, target_hz=target_hz)
        stats_counter[reason] = stats_counter.get(reason, 0) + 1
        if rec is not None:
            eps.append(rec)
            if len(eps) % 100 == 0:
                mb = rstream.offset / (1024 * 1024)
                print(
                    f"  [cfg{cfg_num}] valid={len(eps)}/{seen_eps} tasks={len(set(e['task'] for e in eps))} "
                    f"dl={mb:.0f}MB elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
            if len(eps) % save_every == 0:
                save_canonical_shard(out_path, eps, stats_counter, url, time.time() - t0)

    try:
        with tarfile.open(fileobj=rstream, mode="r|gz") as tf:
            for member in tf:
                if time.time() - t0 >= max_seconds or len(eps) >= max_episodes:
                    break
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 3:
                    continue
                ep_name = parts[1]
                fname = parts[-1]
                if cur_ep is not None and ep_name != cur_ep:
                    _flush_cur()
                    cur_blobs = {}
                    if time.time() - t0 >= max_seconds or len(eps) >= max_episodes:
                        cur_ep = None
                        break
                cur_ep = ep_name
                if fname in NEEDED_FILES:
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        cur_blobs[fname] = fobj.read()
        _flush_cur()
    except Exception as e:
        print(f"  [cfg{cfg_num}] stream ended ({type(e).__name__}: {e}); flushing {len(eps)} valid episodes", flush=True)

    save_canonical_shard(out_path, eps, stats_counter, url, time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=int, required=True, help="RH20T config index 1..7")
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=1200)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()
    stream_cfg(
        args.cfg,
        Path(args.out_npz),
        args.max_episodes,
        args.max_seconds,
        args.target_hz,
        chunk_mb=args.chunk_mb,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
