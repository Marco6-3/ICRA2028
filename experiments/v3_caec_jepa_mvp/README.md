# V3 MVP：`CAEC-JEPA` 公开数据集（ManiWAV）严格审计验证

> **目标**：在 `NewGripper` 硬件样机采数就绪前，基于最同构的公开手持接触声振操作数据集 **`ManiWAV`（CoRL 2024，接触麦克风 + 6-DOF 连续位姿动作）**，在 Tailscale 远程计算节点 `nvidia-desktop`（Jetson Orin CUDA）上完成 V3 算法的首轮可否证验证。

## 文件说明

- [inspect_and_cache_maniwav.py](inspect_and_cache_maniwav.py)：解析 ManiWAV `replay_buffer.zarr.zip`，提取严格因果多频子步声振特征（双通道 log-Mel + RMS + 频谱通量）与局部坐标系相对动作，按 **episode 级严格划分 70/15/15 Train/Val/Test**，并仅用 Train 统计量归一化。
- [models_caec_jepa.py](models_caec_jepa.py)：实现 `CAEC-JEPA` 完整模型、严格纯接触编码器（零本体捷径）、慢上下文编码器、零动作恒等分解预测器（$g_\phi(\cdot, \mathbf{0}) \equiv \mathbf{0}$）、`[CTX, STATE, EVENT]` 3-Token 接口，以及同容量对照基线（`B0_ProprioOnly`、`B1_RawEnd2End`、`B2_MAE`、`B3_VanillaJEPA` 与组件消融）。
- [run_v3_mvp_benchmark.py](run_v3_mvp_benchmark.py)：端到端执行单元审计（前缀因果、零动作恒等、梯度流）、表征预训练、防坍缩几何与纯接触冻结探针、Codex 式“瞬时声振匹配 / 历史路径相反”非马尔可夫压测、速度控制惊奇度突变检测，以及 `10% / 30% / 100% × 3 seeds` 下游少样本 Action-Chunk BC 基准（含未见频谱传递函数偏移 OOD 测试）。

## 远程运行命令（`nvidia-desktop`）

```bash
cd ~/caec_jepa_v3
python3 src/inspect_and_cache_maniwav.py \
  --zarr-zip data/maniwav_flip/replay_buffer.zarr.zip \
  --out-cache data/maniwav_flip/cache_sub4.npz \
  --substeps 4 --n-mels 32 --n-fft 512 --sample-rate 48000

python3 src/run_v3_mvp_benchmark.py \
  --cache data/maniwav_flip/cache_sub4.npz \
  --out-dir outputs/maniwav_flip_v3 \
  --pretrain-steps 1200 --bc-epochs 28 --seeds 0 1 2
```
