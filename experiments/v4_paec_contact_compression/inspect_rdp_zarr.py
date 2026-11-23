#!/usr/bin/env python3
"""
Inspect Reactive Diffusion Policy (RDP) zarr.zip archives (peel / wipe / lift)
and build clean, strictly episode-split caches for V4 PAEC experiments.
"""

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """Convert (..., 6) 6D rotation representation to (..., 3, 3) rotation matrix."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


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


def open_rdp_zarr(path: str):
    """Open either a .zarr directory or a .zarr.zip archive (handles nested root folder)."""
    p = Path(path)
    if p.is_dir():
        root = zarr.open(str(p), mode="r")
        return root, None

    with zipfile.ZipFile(str(p), "r") as zf:
        names = zf.namelist()
    prefix = ""
    for n in names:
        if n.endswith("/data/.zgroup"):
            prefix = n[: -len("/data/.zgroup")]
            break
        if n == "data/.zgroup":
            prefix = ""
            break

    store = zarr.ZipStore(str(p), mode="r")
    root = zarr.open_group(store=store, path=prefix, mode="r")
    return root, store


def inspect_one(path: str):
    root, store = open_rdp_zarr(path)
    print(f"\n=== Inspecting {path} ===")
    print("root keys:", list(root.keys()))
    data_grp = root["data"] if "data" in root else root
    meta_grp = root["meta"] if "meta" in root else None

    for k in sorted(data_grp.keys()):
        try:
            arr = data_grp[k]
            print(f"  data/{k:36s} shape={str(arr.shape):22s} dtype={arr.dtype}")
        except Exception as e:
            print(f"  data/{k:36s} <skip: {e}>")

    if meta_grp is not None:
        for k in sorted(meta_grp.keys()):
            arr = np.asarray(meta_grp[k][:])
            print(f"  meta/{k:36s} shape={arr.shape} first5={arr[:5]} last={arr[-1] if len(arr) else None}")

    if "timestamp" in data_grp:
        ts = np.asarray(data_grp["timestamp"][:], dtype=np.float64)
        dt = np.diff(ts)
        dt_pos = dt[(dt > 1e-4) & (dt < 1.0)]
        if len(dt_pos) > 0:
            print(f"  timestamp median dt={np.median(dt_pos)*1000:.2f} ms ({1.0/np.median(dt_pos):.2f} Hz)")

    for wk in ["left_robot_tcp_wrench", "right_robot_tcp_wrench", "left_robot_gripper_force", "left_gripper1_marker_offset_emb"]:
        if wk in data_grp:
            a = np.asarray(data_grp[wk][:], dtype=np.float32)
            print(f"  stats {wk:32s}: min={a.min():.4f}, max={a.max():.4f}, std={a.std():.4f}, nonzero_frac={(np.abs(a)>1e-5).mean():.3f}")

    if store is not None:
        store.close()


def build_cache(path: str, out_cache: str, seed: int = 42, arm: str = "auto"):
    root, store = open_rdp_zarr(path)
    data_grp = root["data"]
    ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_ep = len(ep_ends)
    total_frames = int(ep_ends[-1])

    # Choose active arm (left or right) based on pose/wrench variance if arm=='auto'
    if arm == "auto":
        l_std = float(np.asarray(data_grp["left_robot_tcp_wrench"][:], dtype=np.float32).std()) if "left_robot_tcp_wrench" in data_grp else -1.0
        r_std = float(np.asarray(data_grp["right_robot_tcp_wrench"][:], dtype=np.float32).std()) if "right_robot_tcp_wrench" in data_grp else -1.0
        arm = "left" if l_std >= r_std else "right"
        print(f"Auto-selected arm='{arm}' (left wrench std={l_std:.4f}, right={r_std:.4f})")

    pose = np.asarray(data_grp[f"{arm}_robot_tcp_pose"][:total_frames], dtype=np.float32)
    wrench = np.asarray(data_grp[f"{arm}_robot_tcp_wrench"][:total_frames], dtype=np.float32)
    grip_w = np.asarray(data_grp[f"{arm}_robot_gripper_width"][:total_frames], dtype=np.float32)
    grip_f = np.asarray(data_grp[f"{arm}_robot_gripper_force"][:total_frames], dtype=np.float32)
    if grip_w.ndim == 1:
        grip_w = grip_w[:, None]
    if grip_f.ndim == 1:
        grip_f = grip_f[:, None]

    vel = (
        np.asarray(data_grp[f"{arm}_robot_tcp_vel"][:total_frames], dtype=np.float32)
        if f"{arm}_robot_tcp_vel" in data_grp
        else np.zeros((total_frames, 6), dtype=np.float32)
    )

    # Tactile marker offset embeddings (keep any active non-constant channels)
    tac_parts = []
    tac_keys = []
    for k in sorted(data_grp.keys()):
        if k.startswith(f"{arm}_gripper") and k.endswith("marker_offset_emb"):
            arr = np.asarray(data_grp[k][:total_frames], dtype=np.float32)
            if float(arr.std()) > 1e-5:
                tac_parts.append(arr)
                tac_keys.append(k)
    if len(tac_parts) > 0:
        tactile = np.concatenate(tac_parts, axis=-1).astype(np.float32)
    else:
        tactile = np.zeros((total_frames, 0), dtype=np.float32)

    # Also grab recorded policy action if available
    raw_action = (
        np.asarray(data_grp["action"][:total_frames], dtype=np.float32)
        if "action" in data_grp
        else np.zeros((total_frames, 0), dtype=np.float32)
    )

    # Compute local-frame pose deltas & gripper deltas per episode (strictly no cross-episode bleed)
    # pose is (N, 9): [pos(3), rot6d(6)] or (N, 7)
    assert pose.shape[1] in (9, 7, 6), f"Unexpected pose dim {pose.shape}"
    proprio_state = np.concatenate([pose, grip_w], axis=-1).astype(np.float32)
    proprio_delta = np.zeros((total_frames, 7), dtype=np.float32)
    action_next = np.zeros((total_frames, 7), dtype=np.float32)

    for s, e in zip(ep_starts, ep_ends):
        s, e = int(s), int(e)
        ep_p = pose[s:e, :3]
        ep_g = grip_w[s:e]
        if pose.shape[1] == 9:
            ep_R = rot6d_to_rotmat(pose[s:e, 3:9])
            R_prev_T = np.swapaxes(ep_R[:-1], -1, -2)
            dp_loc = np.einsum("lij,lj->li", R_prev_T, ep_p[1:] - ep_p[:-1])
            dR_loc = np.matmul(R_prev_T, ep_R[1:])
            dr_loc = rotmat_to_axis_angle(dR_loc)
        else:
            dp_loc = ep_p[1:] - ep_p[:-1]
            dr_loc = np.zeros_like(dp_loc)
        dg = ep_g[1:] - ep_g[:-1]
        step_d = np.concatenate([dp_loc, dr_loc, dg], axis=-1).astype(np.float32)
        proprio_delta[s + 1 : e] = step_d
        action_next[s : e - 1] = step_d

    # Combine physical contact observables: [wrench (6), grip_force (1), tactile (D_tac)]
    contact_obs = np.concatenate([wrench, grip_f, tactile], axis=-1).astype(np.float32)
    # Physical mechanics prediction target: [wrench (6), grip_force (1), tactile (D_tac)]
    # Store wrench_dim=6, grip_force_dim=1, tactile_dim=tactile.shape[1]

    # Strict episode-level train / val / test split (70 / 15 / 15)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ep)
    if n_ep >= 10:
        n_val = max(2, int(round(0.15 * n_ep)))
        n_test = max(2, int(round(0.15 * n_ep)))
        n_train = n_ep - n_val - n_test
    else:
        n_val = 1
        n_test = max(1, n_ep // 4)
        n_train = n_ep - n_val - n_test

    train_eps = np.sort(perm[:n_train])
    val_eps = np.sort(perm[n_train : n_train + n_val])
    test_eps = np.sort(perm[n_train + n_val :])

    train_mask = np.zeros(total_frames, dtype=bool)
    for eid in train_eps:
        train_mask[int(ep_starts[eid]) : int(ep_ends[eid])] = True

    def stats(x):
        m = x[train_mask].mean(axis=0).astype(np.float32)
        s = np.maximum(x[train_mask].std(axis=0), 1e-4).astype(np.float32)
        return m, s

    con_mean, con_std = stats(contact_obs)
    prop_mean, prop_std = stats(proprio_state)
    act_mean, act_std = stats(action_next)

    dt_ms = 0.0
    if "timestamp" in data_grp:
        ts = np.asarray(data_grp["timestamp"][:total_frames], dtype=np.float64)
        dt = np.diff(ts)
        dt_pos = dt[(dt > 1e-4) & (dt < 1.0)]
        if len(dt_pos) > 0:
            dt_ms = float(np.median(dt_pos) * 1000.0)

    out_p = Path(out_cache)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": str(path),
        "arm": arm,
        "n_episodes": int(n_ep),
        "total_frames": int(total_frames),
        "dt_ms": round(dt_ms, 2),
        "fps": round(1000.0 / dt_ms, 2) if dt_ms > 0 else 0.0,
        "wrench_dim": 6,
        "grip_force_dim": 1,
        "tactile_dim": int(tactile.shape[1]),
        "tactile_keys": tac_keys,
        "contact_dim": int(contact_obs.shape[1]),
        "proprio_dim": int(proprio_state.shape[1]),
        "act_dim": 7,
        "raw_action_dim": int(raw_action.shape[1]),
        "n_train_eps": int(len(train_eps)),
        "n_val_eps": int(len(val_eps)),
        "n_test_eps": int(len(test_eps)),
    }
    np.savez(
        out_p,
        contact_obs=contact_obs,
        wrench_raw=wrench,
        proprio_state=proprio_state,
        proprio_delta=proprio_delta,
        action_next=action_next,
        vel_raw=vel,
        raw_action=raw_action,
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
        meta_json=np.array(json.dumps(meta)),
    )
    if store is not None:
        store.close()
    print(f"Saved cache {out_p}: {meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", type=str, required=True)
    parser.add_argument("--out-cache", type=str, default="")
    parser.add_argument("--arm", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inspect_one(args.zarr)
    if args.out_cache:
        build_cache(args.zarr, args.out_cache, seed=args.seed, arm=args.arm)


if __name__ == "__main__":
    main()
