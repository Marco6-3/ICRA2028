# Stage 2P：MCF-Mamba 离线架构对照

## 用户指令与实验

2026-10-07：在现有 TaF Now-casting 管线上将两层 GRU 替换为两层因果 Mamba-1，先进行离线验证。

- Scalar / MCF8，连续历史 T=50/100，种子7/19/31，共24组模型。
- 标签仍为当前 ATI `[Fx,Fy,Mx,My]`；主要观察侧向力矩 RMSE，另报切向力 RMSE。
- GRU 47,460 参数；Mamba 47,978 参数。保持原有 token MLP、pool、FP32、优化、验证选模与源划分。
- Mamba：d_model50、两层 residual/RMSNorm、expand2、d_state16、d_conv4，新增32→50投影和50→4输出头。参数量级匹配，参数总量相差1.09%。
- 使用原生 Windows 可运行的 `mambapy==1.2.0` PyTorch parallel scan + recurrent step；没有调用官方融合 CUDA 内核。
- T100需要更长完整历史，统一筛选公共锚点并重跑GRU T50/T100：训练6,065、验证739、测试9,867窗口，源数量10/2/17。
- 不改变接触活跃筛选阈值，不删帧拼接；只筛窗口末帧。标签和输入校准沿用原流程。
- batch1使用真实连续帧携带状态，测量同步墙钟延迟P50/P95/P99、最大值和低于1ms比例；输入已在GPU，不含传感器和数据传输。

## 实现和核验

入口为 `experiments/contact_operator/run_stage2p_mamba.py`，复用旧 `run_stage2p_nowcast.py` 的数据与训练函数，旧文件保持不变。

训练前验证并行/顺序scan的前向和梯度，整段/逐帧输出一致性及因果性。另复制相同权重到 Transformers 4.57.6 MambaMixer 参考实现进行独立检查。模型训练后验证seed7权重的CPU冷加载和流式输出一致性；重新读取预测文件复算全部模型指标。

## 待检验内容

1. 在同样历史、数据与参数量级下，Mamba是否比GRU降低当前侧向力矩RMSE？
2. T100相对T50是否进一步改善，是否扩大MCF相对Scalar的优势？
3. 本机实际后端的batch1流式延迟能否低于1ms，尾延迟如何？

长程记忆更好、GRU梯度或容量瓶颈均作为待验证解释。本轮不将固定状态大小直接写成优于GRU，也不将PyTorch后端速度当成官方融合内核速度。

Stage 1及此前实验全部保留。24组训练和指标复算已完成。

## 实测结论

- MCF T=50：GRU力矩RMSE 0.0424707，Mamba 0.0454305；Mamba降幅 -6.97%。Mamba−GRU MSE的95%源配对区间 [0.000166158, 0.000318061]。
- MCF T=100：GRU力矩RMSE 0.0428548，Mamba 0.0442267；Mamba降幅 -3.20%。Mamba−GRU MSE的95%源配对区间 [3.08556e-05, 0.00018242]。
- MCF-Mamba从T50增加到T100：侧向力矩RMSE降幅 2.65%；MSE差的95%区间 [-0.00015035010611766259, -3.4667023653894915e-05]。
- 这些结果对应固定小模型、既定数据划分和沿用的GRU优化超参数；本轮完成架构替换对照，不是Mamba超参数最优性搜索。
- 原生Mamba eager P95为 1.944–2.077 ms；同权重CUDA Graph P95为 0.145–0.153 ms。四组各1,000帧Graph测量均低于1ms，测量范围是GPU驻留输入的模型推理；GRU同样优化后更快，未得到Mamba速度优于GRU的结论。

完整报告：[实验报告](../experiments/contact_operator/outputs/stage2p_mamba_20261007/实验报告.md)。模型权重和逐窗口预测已随报告归档在同目录的 `GRU/`、`Mamba/` 子目录；`manifest.json` 记录原始文件校验值。归档辅助整理脚本保留运行时本地路径，正式训练复现入口及参数见归档 `README.md`。
