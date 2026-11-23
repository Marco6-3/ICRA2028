#!/usr/bin/env python3
"""
Inspect ManiWAV zarr.zip dataset and build a strictly causal, multi-rate
contact-audio + proprioception cache for V3 CAEC-JEPA MVP verification.

Strictly enforces:
1. Episode-level train / val / test split (zero cross-split frame leakage).
2. Strict causal STFT/log-Mel extraction (no lookahead padding across future frames).
3. Normalization statistics computed strictly on TRAIN episodes only.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import zarr


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle vectors to (..., 3, 3) rotation matrices."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-6
    safe_theta = np.where(small, 1.0, theta)
    k = aa / safe_theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zeros, -kz, ky], axis=-1),
            np.stack([kz, zeros, -kx], axis=-1),
            np.stack([-ky, kx, zeros], axis=-1),
        ],
        axis=-2,
    )
    I = np.broadcast_to(np.eye(3, dtype=aa.dtype), K.shape)
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    # Small-angle fallback: I + K_aa
    ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2]
    K_aa = np.stack(
        [
            np.stack([zeros, -az, ay], axis=-1),
            np.stack([az, zeros, -ax], axis=-1),
            np.stack([-ay, ax, zeros], axis=-1),
        ],
        axis=-2,
    )
    R_small = I + K_aa
    return np.where(small[..., None], R_small, R)


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (first two columns flattened)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert (..., 3, 3) rotation matrix to (..., 3) axis-angle."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = np.stack([rx, ry, rz], axis=-1)
    sin_t = np.sin(theta)[..., None]
    small = theta[..., None] < 1e-4
    scale = np.where(small, 0.5, theta[..., None] / np.where(small, 1.0, 2.0 * sin_t))
    return r * scale


def build_mel_filterbank(n_fft: int, sample_rate: int, n_mels: int, fmin: float = 80.0, fmax: float = 16000.0) -> np.ndarray:
    """Pure-NumPy triangular Mel filterbank matrix of shape (n_fft // 2 + 1, n_mels)."""
    fmax = min(fmax, sample_rate * 0.49)

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for i in range(n_mels):
        f_l, f_c, f_r = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - f_l) / max(f_c - f_l, 1e-6)
        right = (f_r - freqs) / max(f_r - f_c, 1e-6)
        fb[:, i] = np.maximum(0.0, np.minimum(left, right))
    # Area-normalize each filter
    fb /= np.maximum(fb.sum(axis=0, keepdims=True), 1e-6)
    return fb


def extract_causal_audio_substeps(
    ep_audio: np.ndarray,
    substeps: int,
    mel_fb: np.ndarray,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal sub-step log-Mel + broadband RMS extraction for one episode.

    Args:
        ep_audio: shape (T, C, block_size), float32 waveform chunks per video frame.
        substeps: number of causal sub-steps per macro frame (e.g. 4).
        mel_fb: shape (n_fft // 2 + 1, n_mels).
        n_fft: causal window length in samples.

    Returns:
        feats: shape (T, substeps, C * (n_mels + 2)), strictly causal (each sub-step
               uses ONLY samples up to the end of that sub-step; zero future samples).
        rms: shape (T, substeps), broadband RMS energy across channels.
    """
    T, C, B = ep_audio.shape
    hop = B // substeps
    assert hop > 0 and hop * substeps <= B, f"Invalid block_size={B} for substeps={substeps}"
    # Concatenate episode audio along time: (C, T * B)
    # Note: if B is not evenly divisible by substeps, we place sub-step ends evenly in [0, B]
    wave = ep_audio.transpose(1, 0, 2).reshape(C, T * B).astype(np.float32)
    # Left-pad strictly on the past side only -> zero lookahead!
    padded = np.pad(wave, ((0, 0), (n_fft, 0)), mode="constant", constant_values=0.0)
    win = np.hanning(n_fft).astype(np.float32)[None, None, :]  # (1, 1, n_fft)

    total_ticks = T * substeps
    # End indices in original wave for each tick (1-based slice end in wave -> shifted by +n_fft in padded)
    # For frame t and substep s in 0..substeps-1, end sample inside frame is round((s+1)*B/substeps)
    sub_ends = np.array([int(round((s + 1) * B / substeps)) for s in range(substeps)], dtype=np.int64)
    tick_ends = (np.arange(T, dtype=np.int64)[:, None] * B + sub_ends[None, :]).reshape(-1)

    # Gather causal windows: padded[:, end : end + n_fft] covers wave[end - n_fft : end]
    offsets = np.arange(n_fft, dtype=np.int64)[None, :]  # (1, n_fft)
    idx = tick_ends[:, None] + offsets  # (total_ticks, n_fft), values in [0, T*B + n_fft]
    # Because padded has n_fft zeros prepended, wave[0] is at padded[n_fft],
    # so wave[end - n_fft : end] is padded[end : end + n_fft].
    windows = padded[:, idx]  # (C, total_ticks, n_fft)
    windows = windows - windows.mean(axis=-1, keepdims=True)
    rms_per_ch = np.sqrt(np.mean(windows ** 2, axis=-1) + 1e-10)  # (C, total_ticks)
    log_rms = np.log(rms_per_ch + 1e-6)  # (C, total_ticks)

    spec = np.fft.rfft(windows * win, n=n_fft, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2) / n_fft  # (C, total_ticks, n_fft//2+1)
    mel = np.matmul(power, mel_fb)  # (C, total_ticks, n_mels)
    log_mel = np.log(mel + 1e-6)  # (C, total_ticks, n_mels)

    # Causal spectral flux (L2 norm of positive frame-to-frame log-mel rise)
    flux = np.zeros((C, total_ticks), dtype=np.float32)
    diff_mel = np.maximum(log_mel[:, 1:, :] - log_mel[:, :-1, :], 0.0)
    flux[:, 1:] = np.sqrt(np.mean(diff_mel ** 2, axis=-1) + 1e-10)

    # Stack per channel: [log_mel (n_mels), log_rms (1), flux (1)] -> (total_ticks, C*(n_mels+2))
    ch_feats = np.concatenate(
        [log_mel, log_rms[..., None], flux[..., None]], axis=-1
    )  # (C, total_ticks, n_mels + 2)
    feats = ch_feats.transpose(1, 0, 2).reshape(T, substeps, C * (mel_fb.shape[1] + 2)).astype(np.float32)
    rms = rms_per_ch.mean(axis=0).reshape(T, substeps).astype(np.float32)
    return feats, rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-zip", type=str, required=True)
    parser.add_argument("--out-cache", type=str, required=True)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--n-mels", type=int, default=32)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Opening {args.zarr_zip} ...")
    store = zarr.ZipStore(args.zarr_zip, mode="r")
    root = zarr.group(store=store)

    data_keys = list(root["data"].keys())
    meta_keys = list(root["meta"].keys())
    data_summary = {}
    for k in data_keys:
        try:
            arr = root["data"][k]
            data_summary[k] = (arr.shape, str(arr.dtype))
        except Exception as exc:
            data_summary[k] = f"<skipped: {type(exc).__name__}: {exc}>"
    print("data keys:", data_summary)
    print("meta keys:", {k: (root["meta"][k].shape, str(root["meta"][k].dtype)) for k in meta_keys})

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    n_episodes_total = len(episode_ends)
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_lens = episode_ends - ep_starts
    print(
        f"Total episodes: {n_episodes_total}, total frames: {int(episode_ends[-1])}, "
        f"len min/med/max: {int(ep_lens.min())}/{int(np.median(ep_lens))}/{int(ep_lens.max())}"
    )

    # Identify mic keys
    mic_keys = sorted([k for k in data_keys if k.startswith("mic_")])
    assert len(mic_keys) > 0, f"No mic_* keys found in {data_keys}"
    pos_key = "robot0_eef_pos"
    rot_key = "robot0_eef_rot_axis_angle"
    grip_key = "robot0_gripper_width"
    for k in [pos_key, rot_key, grip_key]:
        assert k in data_keys, f"Missing required key {k} in {data_keys}"

    n_ep = n_episodes_total if args.max_episodes <= 0 else min(args.max_episodes, n_episodes_total)
    total_frames = int(episode_ends[n_ep - 1])

    print(f"Loading {n_ep} episodes ({total_frames} frames) of pose, gripper, and {mic_keys} ...")
    pos = np.asarray(root["data"][pos_key][:total_frames], dtype=np.float32)
    rot_aa = np.asarray(root["data"][rot_key][:total_frames], dtype=np.float32)
    grip = np.asarray(root["data"][grip_key][:total_frames], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]

    mics = []
    for mk in mic_keys:
        arr = np.asarray(root["data"][mk][:total_frames], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        mics.append(arr)
    # Stack channels: (N, C, block_size)
    audio_all = np.stack(mics, axis=1)
    block_size = audio_all.shape[-1]
    print(f"Loaded audio_all shape={audio_all.shape}, pos={pos.shape}, rot_aa={rot_aa.shape}, grip={grip.shape}")
    inferred_fps = args.sample_rate / float(block_size)
    print(f"Inferred macro FPS (assuming {args.sample_rate} Hz audio): {inferred_fps:.2f} Hz, "
          f"sub-step clock ({args.substeps}x): {inferred_fps * args.substeps:.2f} Hz")

    # Build Mel filterbank
    n_fft = min(args.n_fft, block_size)
    mel_fb = build_mel_filterbank(n_fft=n_fft, sample_rate=args.sample_rate, n_mels=args.n_mels)

    # Process episode by episode strictly without cross-episode bleed
    ep_starts_sel = ep_starts[:n_ep]
    ep_ends_sel = episode_ends[:n_ep]

    audio_dim = len(mic_keys) * (args.n_mels + 2)
    audio_feats = np.zeros((total_frames, args.substeps, audio_dim), dtype=np.float32)
    audio_rms = np.zeros((total_frames, args.substeps), dtype=np.float32)
    proprio_state = np.zeros((total_frames, 10), dtype=np.float32)  # [pos(3), rot6d(6), grip(1)]
    proprio_delta = np.zeros((total_frames, 7), dtype=np.float32)   # causal [dpos_local(3), drot_local(3), dgrip(1)]
    action_next = np.zeros((total_frames, 7), dtype=np.float32)     # next-step local delta [dp(3), dr(3), dg(1)]

    for e in range(n_ep):
        s, t_end = int(ep_starts_sel[e]), int(ep_ends_sel[e])
        ep_p = pos[s:t_end]
        ep_r_aa = rot_aa[s:t_end]
        ep_g = grip[s:t_end]
        ep_R = axis_angle_to_rotmat(ep_r_aa)  # (L, 3, 3)
        ep_r6 = rotmat_to_rot6d(ep_R)         # (L, 6)
        proprio_state[s:t_end] = np.concatenate([ep_p, ep_r6, ep_g], axis=-1)

        # Relative local-frame deltas between consecutive steps:
        # delta from t-1 to t expressed in frame t-1:
        #   dp_local = R_{t-1}^T (p_t - p_{t-1})
        #   dR_local = R_{t-1}^T R_t -> axis-angle
        R_prev_T = np.swapaxes(ep_R[:-1], -1, -2)
        dp_local = np.einsum("lij,lj->li", R_prev_T, ep_p[1:] - ep_p[:-1])
        dR_local = np.matmul(R_prev_T, ep_R[1:])
        dr_local = rotmat_to_axis_angle(dR_local)
        dg = ep_g[1:] - ep_g[:-1]
        step_delta = np.concatenate([dp_local, dr_local, dg], axis=-1).astype(np.float32)

        # Causal proprio_delta[t] = step_delta executed to reach frame t (0 at t=0)
        proprio_delta[s + 1 : t_end] = step_delta
        # Action executed FROM frame t to t+1 (last frame repeats 0)
        action_next[s : t_end - 1] = step_delta

        # Causal audio substeps
        af, ar = extract_causal_audio_substeps(audio_all[s:t_end], args.substeps, mel_fb, n_fft)
        audio_feats[s:t_end] = af
        audio_rms[s:t_end] = ar

    # Check NaNs/Infs
    for name, arr in [
        ("audio_feats", audio_feats),
        ("proprio_state", proprio_state),
        ("proprio_delta", proprio_delta),
        ("action_next", action_next),
    ]:
        assert np.isfinite(arr).all(), f"Non-finite values detected in {name}!"

    # Strict episode-level train / val / test split (70 / 15 / 15)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ep)
    n_train = max(1, int(round(0.70 * n_ep)))
    n_val = max(1, int(round(0.15 * n_ep)))
    n_test = max(1, n_ep - n_train - n_val)
    if n_train + n_val + n_test > n_ep:
        n_train = n_ep - n_val - n_test
    train_eps = np.sort(perm[:n_train])
    val_eps = np.sort(perm[n_train : n_train + n_val])
    test_eps = np.sort(perm[n_train + n_val :])
    assert len(set(train_eps) & set(val_eps)) == 0
    assert len(set(train_eps) & set(test_eps)) == 0
    assert len(set(val_eps) & set(test_eps)) == 0

    # Compute normalization statistics strictly on TRAIN episodes only
    train_mask = np.zeros(total_frames, dtype=bool)
    for e in train_eps:
        train_mask[int(ep_starts_sel[e]) : int(ep_ends_sel[e])] = True

    aud_train = audio_feats[train_mask].reshape(-1, audio_dim)
    aud_mean = aud_train.mean(axis=0).astype(np.float32)
    aud_std = np.maximum(aud_train.std(axis=0), 1e-4).astype(np.float32)

    prop_mean = proprio_state[train_mask].mean(axis=0).astype(np.float32)
    prop_std = np.maximum(proprio_state[train_mask].std(axis=0), 1e-4).astype(np.float32)

    act_mean = action_next[train_mask].mean(axis=0).astype(np.float32)
    act_std = np.maximum(action_next[train_mask].std(axis=0), 1e-4).astype(np.float32)

    out_path = Path(args.out_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        audio_feats=audio_feats,
        audio_rms=audio_rms,
        proprio_state=proprio_state,
        proprio_delta=proprio_delta,
        action_next=action_next,
        ep_starts=ep_starts_sel,
        ep_ends=ep_ends_sel,
        train_eps=train_eps,
        val_eps=val_eps,
        test_eps=test_eps,
        aud_mean=aud_mean,
        aud_std=aud_std,
        prop_mean=prop_mean,
        prop_std=prop_std,
        act_mean=act_mean,
        act_std=act_std,
        meta_json=np.array(
            json.dumps(
                {
                    "source": str(args.zarr_zip),
                    "n_episodes": int(n_ep),
                    "total_frames": int(total_frames),
                    "mic_keys": mic_keys,
                    "block_size": int(block_size),
                    "sample_rate": int(args.sample_rate),
                    "inferred_fps": float(inferred_fps),
                    "substeps": int(args.substeps),
                    "n_mels": int(args.n_mels),
                    "n_fft": int(n_fft),
                    "audio_dim": int(audio_dim),
                    "n_train_eps": int(len(train_eps)),
                    "n_val_eps": int(len(val_eps)),
                    "n_test_eps": int(len(test_eps)),
                }
            )
        ),
    )
    store.close()
    print(f"Saved cache to {out_path} in {time.time() - t0:.1f}s.")
    print(f"Split episodes: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")


if __name__ == "__main__":
    main()
