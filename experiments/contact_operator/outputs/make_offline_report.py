"""Generate the Chinese evidence report from completed experiment files."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent
folder=root/'offline_validation_tuned'
read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
summary=read('summary.json');results=read('results.json');timing=read('cpu_descriptor_timing.json')
checks=read('implementation_checks.json');protocol=read('descriptor_protocol.json')
audit=json.loads((root/'offline_validation'/'data_audit.json').read_text(encoding='utf-8'))
split=read('split.json');chosen=read('selected_hyperparameters.json')
labels={'State':'仅本体状态','Scalar':'本体状态 + 总响应','Physics11D':'本体状态 + Physics11D',
        'Raw':'本体状态 + 稠密触觉','Learned11D':'本体状态 + Learned11D'}
table=[]
for item in summary:
    name=item['representation'];fit=next(r['fit'] for r in results if r['family']=='MLP' and r['representation']==name and r['fraction']==1)
    ci=item['paired_delta_nmse_CI95']
    table.append(f"| {labels[name]} | {item['mean_macro_nmse']:.4f} | {item['sd_across_training_seeds']:.4f} | {item['relative_change_vs_state_percent']:+.1f}% | [{ci[0]:.4f}, {ci[1]:.4f}] | {fit['parameter_count']:,} |")
curve=[]
for fraction in [.25,.5,1]:
    values=[]
    for name in labels:
        rows=[r for r in results if r['family']=='MLP' and r['representation']==name and r['fraction']==fraction]
        values.append(f"{np.mean([r['test']['macro_nmse'] for r in rows]):.4f}")
    curve.append('| '+str(round(44*fraction))+' | '+' | '.join(values)+' |')
ridge=[]
for r in results:
    if r['family']=='Ridge':ridge.append(f"| {labels[r['representation']]} | {r['test']['macro_nmse']:.4f} | {r['alpha']} |")
physics=next(s for s in summary if s['representation']=='Physics11D')
if physics['paired_delta_nmse_CI95'][1]<0:
    judgment='在本次设定中，Physics11D 相对本体状态读出出现改善，但仍属于同任务、同一划分的探索性证据。'
elif physics['relative_change_vs_state_percent']>=0:
    judgment='本次没有发现 Physics11D 相对本体状态的动作预测优势；平均误差反而更高。'
else:
    judgment='Physics11D 的平均误差稍低，但配对区间包含零，目前不足以确认增量价值。'
text=f'''# LeFlexiTac 离线验证：首轮结果与边界

日期：2026-10-06。状态：已实际运行的探索性实验；不是闭环控制或完整 Stage 0/1 的通过证明。

**{judgment}** 不能据此宣称最小充分状态、完整保留微滑移信息、优于公开 token 路径或真实反应速度。按用户要求，本次没有加入 token 基线。

## 数据与实际完成的工作

数据：[Tna001/tactile_test_tube_pyflexitac](https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac)，版本 `ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4`。共 64 条试验、75,041 帧、12×32 阵列，10 个 Parquet 文件。全部下载文件与对应公开 LFS SHA-256 校验一致。数据卡标注 Apache-2.0。

保存值范围 0～1，全零帧约 {audit['zero_mass_fraction']*100:.1f}%，相邻完全相同触觉帧约 {audit['adjacent_identical_fraction']*100:.1f}%。保存时间轴约 30 Hz；时间戳几乎等于 frame_index/30，不能用于证明真实传感器采样抖动或端到端延迟。上游校准、归一化及噪声处理语义尚未完整核查，因此称为“保存的触觉响应”，不称原始 ADC 或绝对压力。

固定随机划分 44 条训练、10 条验证、10 条测试，整条试验不跨集合。未获得物体、材质或采集批次标签；这是同任务 episode 留出，不是 OOD。

完成内容：因果预处理检查、8 个静态量和 3 个差分量、5 类输入的 MLP 比较、4 类输入的 Ridge 诊断、11/22/44 条训练试验的子集扫描、3 个训练随机种子、验证集超参数选择、保存预测及检查点。

## 预测任务和公平比较

目标为 `action[t+5] - observation.state[t]`，即约 167 ms 后的六个位置指令相对当前位置的偏差。它是操作员命令的离线拟合，不是受力真值、稳定性真值或最佳纠偏动作。

所有模型共享本体输入：当前位置和上一帧位置变化率，共 12 维。触觉模型另共享当前 / 上一帧接触有效性标记（2 维），模型名单中的 11D 不包括这两个辅助标记。

| 表示 | 额外触觉输入 | 网络 |
| --- | --- | --- |
| State | 无 | 64→64→6 的 MLP 读出 |
| Scalar | 对数总响应及相邻帧差，共 2 维 | 同宽读出 |
| Physics11D | 8 个静态量 + 3 个相邻帧差 | 同宽读出 |
| Raw | 当前及上一帧预处理后阵列，共 768 维 | 同宽读出 |
| Learned11D | 相同 768 维阵列经 768→64→11 编码器 | 11D 接相同读出，联合监督训练 |

未使用图像或 Mamba。Raw 与 Learned11D 都能访问上一帧，避免 Physics 差分独占历史。相同读出宽度不等于总参数量相同；表中报告实际参数量。Raw 采用小 MLP，不是经过完整调优的 CNN 或公开 dense-token policy；结果不能推广到所有高维触觉模型。

## 明确的算子实现

以仓库的通道顺序 `[P', cx, cy, A, σ1, σ2, ox, oy, ΔP', Δcx, Δcy]` 实现；与你补充文本的静态顺序不同只是一种排列。

- 上游数据已经是 0～1 响应，因此不再次使用元数据中的 ADC 阈值 25，也不重复扣除 baseline。
- 所有触觉输入共享因果 EMA（α=0.5）和固定截断（响应 ≤0.01 置零）。矩和面积从截断后的同一组非负权重计算。
- `P_ref={protocol['P_ref']:.6f}`，由 44 条训练试验的正总量固定，测试不更新。没有在线 Z-score。
- 两轴归一化到 [−1,1]；计算的是传感器坐标几何，不是保留真实长宽比的物理尺寸。
- 采用正总量精确分母与各向异性置信度；P≤1e−6 时填零，并另给有效性标记。
- 三个动态项按照你补充的公式使用相邻帧差，不称每秒变化率。重心差只有前后都有效时才保留。
- 每条试验独立初始化滤波。改变未来输入不能改变前缀输出的检查通过。
- 读出训练 / 评价取每第三帧，但所有输入滤波、相邻帧差和五帧未来标签都先在保存的 30 Hz 时间轴计算；共用相同读出行。

向量化静态矩与独立标量实现的最大差异为 `{checks['vectorized_vs_scalar_max_error']:.3g}`，零接触与全部 Physics 特征均无 NaN/Inf。这验证了实现的一致性，没有独立事件标签，不能证明真实 onset / release 的物理准确性。

## 训练、选择和结果指标

先运行了 80 轮上限的探索性试验。45 次拟合中有 25 次最佳轮数≥75，因此保留初轮文件，并增加到最多 240 轮、早停耐心 16。每种输入在相同四组学习率 / 正则候选中按验证集选配置，再固定配置运行 3 个随机种子。候选搜索不计算测试成绩。

候选为学习率 0.001 / 0.0003、weight decay 0.0001 / 0.01。最终配置见 `selected_hyperparameters.json`。输入和目标的标准化均只用本次选定训练子集；Physics 的 P_ref 沿用 44 条训练的固定标定，子集扫描也沿用完整训练预算选择的超参数。因此低样本曲线是**固定标定与配置下的训练子集扫描**，不是严格仅允许 11 或 22 条数据参与整个开发流程的结论。

主指标是测试 episode 等权平均的 normalized MSE，越低越好。每个关节残差除以完整训练集该关节目标的标准差，随后求平方均值。该完整训练集尺度只用于统一评价单位，不供少数据模型拟合。位置保持基线（未来指令预测为当前状态）的指标为 {results[0]['test']['macro_nmse']:.4f}。

下表是完整 44 条训练的结果。±变化以 State 为参照，正数代表更差。配对 95% 区间以同一测试试验的误差差值（候选−State）计算，先平均三个训练种子，再对 10 个试验做 5,000 次 cluster bootstrap。

| 输入 | 平均 normalized MSE | 种子间标准差 | 相对 State 变化 | 配对差值 95% 区间 | 参数量 |
| --- | ---: | ---: | ---: | --- | ---: |
{chr(10).join(table)}

这些区间仅描述当前 10 条试验，不是跨任务泛化或独立确认性检验。初轮已看过相同测试集合，后续仅按验证集调参，但整体仍必须标为探索性。不能把帧当作几万个独立试验。

## 子集扫描

每个预算取嵌套的完整训练试验，报告三个随机种子的平均误差。

| 训练试验数 | State | Scalar | Physics11D | Raw | Learned11D |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(curve)}

![训练子集扫描](data_efficiency.png)

误差条是训练随机种子的标准差，不是独立试验的置信区间。Physics 比某个 Raw / Learned 实现更好，若同时不能改善 State-only 或 Scalar，就不足以支持“空间触觉具有增量价值”的主张。

线性 Ridge 诊断（alpha 仅按验证集选择）：

| 输入 | normalized MSE | alpha |
| --- | ---: | ---: |
{chr(10).join(ridge)}

Ridge 与 MLP 的结论可能不同，说明结果依赖读出能力；不能挑选单个读出来宣布算子成立。

## 本机计算耗时

CPU：AMD Ryzen 7 H 260。GPU 训练：RTX 5060 Laptop。预热 200 帧后，连续测量 1,000 帧 Python CPU 的 EMA、截断、8 静态量、3 差分量和状态更新：P50 **{timing['p50_ms']:.4f} ms**，P95 **{timing['p95_ms']:.4f} ms**。

这是本机计算路径的描述性测量，不含传感器、串口、策略模型、执行器，不是事件到动作生效的闭环延迟。未测硬件噪声、漂移、事件重复性或 100 Hz 闭环。

## 对你补充推导的修正

1. 更准确的名称是“基于零至二阶空间矩、有效面积和相邻帧差的解析描述子”。保留有限矩不等于建立可重建场的级数展开。
2. “最小充分物理状态”需要特定任务、观测 / 动力学模型及充分性证明；目前没有。上次精确反例已经证明不同压力图可得到相同 11D。
3. √λ 是加权标准差，不直接等于真实接触斑半轴。例如均匀填充椭圆的几何半轴是 2√λ；对一般压力场不存在统一边界解释。
4. 双角方向编码是协方差的各向异性表示；它的形式与对称二阶张量变换一致，但协方差不是应力张量，不必称为莫尔圆力学投影。
5. Δc 表示接触响应重分布，可由滚动、倾斜、加载位置变化等造成，不能仅据此识别微滑移。Mamba 隐状态也不能预先称为完整时空导数。
6. 算子具有阈值、滤波和标定参数；可称“不需要学习权重”，不宜称“完全无参、零延迟”。单帧阵列处理随 taxel 数量约 O(N)；固定隐藏维度的递推对历史长度可为 O(1)，并不等于硬件耗时恒定或足够小。
7. 创新、首次、<0.1 ms、完整保留控制必要信息和实机收益，必须分别由文献、测量与任务证据支持。本轮没有验证 token、VLA、阻抗控制或双闭环架构。

## 下一步判断

当前不依据这组结果启动 Mamba / VLA。应首先核查保存响应的预处理链；设计受控的“总量近似不变、空间分布变化”数据，并提供独立接触位置 / 偏心扰动或所需恢复方向标签。只预测操作员位置命令，容易被本体状态解释，未必能诊断空间接触表示。

若目标任务仍没有几何增量，应缩小到总响应或修改表示；不要因为 Physics 胜过一个过拟合的 Raw 网络就宣布贡献。是否上实机应以受控 Stage 0 与表示证据决定。

## 文件与复现

- `results.json`：逐模型、训练预算、种子、测试试验与 MAE 的完整结果，包含负结果。
- `validation_tuning.json`、`selected_hyperparameters.json`：调参候选与选择依据。
- `split.json`、`descriptor_protocol.json`：独立试验划分与算子定义。
- `predictions_*.npz`、`*_seed*.pt`：完整训练预算的预测和检查点。
- `data_efficiency.png`：可直接用于讨论的探索性结果图。
- `implementation_checks.json`、`cpu_descriptor_timing.json`：实现检查和计算测量。

从任务目录运行 `python outputs/run_offline_validation.py` 下载数据、检查并复现初轮，再运行 `python outputs/tune_offline_validation.py` 复现主要实验。依赖 NumPy、PyArrow、PyTorch、scikit-learn、Matplotlib。第一次下载只取约 {sum(m['bytes'] for m in audit['files'])/1e6:.2f} MB 表格数据，无需视频或 LeRobot 全套环境。数据引用与许可请随实验保留。
'''
(folder/'离线验证报告.md').write_text(text,encoding='utf-8')
print('REPORT_WRITTEN',len(text))
