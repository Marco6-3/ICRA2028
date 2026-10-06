# 数据来源、许可与实验边界

本目录使用公开数据的压力／响应表格和外部力测量进行探索性实验；不复刻原论文的完整策略，也不把作者报告当成本项目实测。

| 数据 | 固定版本 | 用途与来源 |
| --- | --- | --- |
| HTT | b3d6f0a275dfc949cbe6c100e4ca767e4ad8bf59 | [数据集与格式](https://huggingface.co/datasets/AllenBi21/HTT-dataset/blob/b3d6f0a275dfc949cbe6c100e4ca767e4ad8bf59/FORMAT.md)；12 条 Xela 记录，72 通道 L1 响应标量筛查，不能冒称二维法向阵列验证 |
| LeFlexiTac test tube | ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4 | [固定数据版本](https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac/tree/ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4)；64 episode、75,041 帧，12×32 保存响应、约 30 Hz；不是已核实原始 ADC 或 100 Hz 采集流 |
| TaF | 239c8ee8156bf11eb6d6c11c004f9a8721528d9d | [数据卡](https://huggingface.co/datasets/jiamig/taf-dataset/blob/239c8ee8156bf11eb6d6c11c004f9a8721528d9d/README.md)、[采集论文](https://arxiv.org/html/2601.20321v1#S3)；12×12 阵列与独立 ATI 六维力／力矩，约 30 Hz |

HTT 数据卡／格式标注 CC-BY-NC-4.0，本提交只保留引用、聚合指标与下载脚本，不分发其原始 NPZ。TaF 数据卡标注 MIT；提交的 `stable_load_real_windows.npz` 与 `load_matched_real_pairs.npz` 是从 TaF 筛查得到的派生数值样本，来源、基线、筛查门槛、原始记录行号与 SHA 均保留，使用时请同时引用上述数据集与原论文。

LeFlexiTac 原始表格不随提交分发；下载与后续使用时以固定版本对应的作者数据卡、许可和项目说明为准。本提交不凭“公开下载”推断其可任意再分发。

TaF 引用：Huang, Yuzhe; Lin, Pei; Li, Wanlin; Li, Daohan; Li, Jiajun; Jiang, Jiaming; Xiao, Chenxi; Jiao, Ziyuan. *TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation*. arXiv:2601.20321, 2026。

TaF 数据卡将阵列称为 piezoelectric，论文称 calibrated piezoresistive；该差异未独立解决。阵列坐标与 ATI 安装偏置、实际绝对单位未核实，因此仅报告保存单位下 `[-Ty/Fz, Tx/Fz]` 的离线预测误差。源记录编号不自动等于不同物体、材质或独立试验批次。
