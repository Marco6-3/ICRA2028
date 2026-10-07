# TaF Mamba / GRU 离线对照复现

完整结果见 `实验报告.md`，原始指标见 `results.json`，配对区间见 `summary.json`。

依赖：Python3.11、torch2.11.0+cu128、numpy、pyarrow、threadpoolctl、mambapy1.2.0；独立参考检查另需transformers4.57.6。训练设备为RTX5060 Laptop GPU，FP32且TF32关闭。

```powershell
python run_stage2p_mamba.py --cache-root '<含taf目录的数据根目录>' --out '<新的空输出目录>'
```

数据需包含obj7至obj54 parquet以及对应objN_listing.json，脚本逐文件验证LFS SHA256。必须选择新目录，防止覆盖旧实验。

`GRU/` 和 `Mamba/` 内每组保存 `.pt` 与 `_pred.npy`。预测行顺序对应 `examples.npz` 的 `test` 索引。加载自有检查点时使用 `torch.load(..., weights_only=False)`；根据架构选择脚本中的 `GRUProbe` 或 `MambaProbe`，权重为 `state_dict`。归一化和反归一化参数保存在检查点。

`protocol.json` 记录脚本和后端SHA256；`implementation_checks.json`、`hf_reference_checks.json`、`artifact_verification.json` 记录数值核验。流式单步调用 `step(net, frame, state)` 并在episode开始清空state。不能在独立窗口间误携带状态。
