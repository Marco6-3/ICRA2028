# 接触算子：数学检查与公开数据离线验证

更新：2026-10-06。已完成数学检查、HTT 标量筛查、LeFlexiTac 动作预测、TaF 数据筛查及空间状态诊断。**没有可靠证明完整 11D 的增量价值；Stage 1 尚未通过。** 未训练 Mamba、VLA、dense tactile token 基线，也没有实机闭环结果。

## 建议阅读顺序

1. [成果与决策总览](../../research/OFFLINE_RESULTS.md)：最新结果与继续条件。
2. [数学检查与信息丢失反例](outputs/算子初步验证.md)。
3. [LeFlexiTac 主要离线报告](outputs/offline_validation_tuned/离线验证报告.md)。
4. [TaF 训练前数据筛查](outputs/spatial_contact_screen/空间接触数据筛查与下一轮方案.md)。
5. [TaF 空间诊断报告](outputs/spatial_validation_expanded/空间算子离线验证报告.md)。

## 目录与证据

| 路径 | 内容 |
| --- | --- |
| outputs/*.py | 原实验脚本，保留固定版本数据下载、校验和、因果预处理与读出代码 |
| outputs/operator_validation_results.json | 数学检查及限定范围的 HTT 标量筛查 |
| outputs/offline_validation/ | LeFlexiTac 初轮协议、结果、划分与图表；探索性试跑，不能与调优版混为一轮 |
| outputs/offline_validation_tuned/ | 验证集调优后的主要结果、3 种子小型检查点、实现检查与 CPU 计算计时 |
| outputs/spatial_contact_screen/ | 六源记录筛查、316 个真实稳载窗口、100 对载荷匹配样本及全部检索审计 |
| outputs/spatial_validation/ | 原 TaF 划分与失败审计：原定 4 个测试记录全部未通过校准 |
| outputs/spatial_validation_expanded/ | 修订后的完整结果、逐源误差、17 源记录留出、阈值敏感性、窗口动态诊断与模型 |

`work/` 是运行后下载／生成的缓存，不进入 Git。大型 `predictions_*.npz` 和 `*_prediction.npz` 未随仓库提交，可按下述命令生成；小型线性权重、主要神经检查点与 TaF 筛查的两个派生样本包已提交。报告里的文件清单描述完整原实验产物，不意味着每个逐帧文件都在 Git 中。

## 安装与运行位置

Python 3.11。首次运行需要网络下载固定 SHA 下的公开表格，不下载视频。从仓库根目录执行：

```bash
cd experiments/contact_operator
python -m venv .venv
```

激活虚拟环境后安装 `python -m pip install -r requirements.txt`。TaF 神经训练脚本目前直接使用 CUDA；本轮实测环境为 PyTorch 2.11.0+cu128 与 RTX 5060 Laptop。请按实际系统安装兼容的 PyTorch GPU wheel；CPU 机器可以检查存档结果与运行数学检查，完整 TaF 神经训练需修改设备路径后另行记录，不能当成本轮原样复现。

## 复现命令

以下命令均在 `experiments/contact_operator` 内运行。

数学检查与 HTT 限定筛查：

```bash
python outputs/validate_contact_operator.py
```

LeFlexiTac：初轮 → 验证集调优 → 报告。`--prepare-only` 只下载／审计，不代替产生初轮训练结果。

```bash
python outputs/run_offline_validation.py --prepare-only
python outputs/run_offline_validation.py
python outputs/tune_offline_validation.py
python outputs/make_offline_report.py
```

TaF：六记录筛查 → 导出真实样本 → 完整源记录留出 → 报告与模型冷加载检查。

```bash
python outputs/screen_spatial_contact_data.py
python outputs/export_spatial_windows.py
python outputs/run_spatial_validation.py --prepare-only
python outputs/run_spatial_validation.py
python outputs/report_spatial_validation.py
```

默认脚本会重写相应结果文件。要进行不覆盖存档的完整重跑，把本目录复制到独立目录后执行；TaF 脚本若发现既存 `neural_tuning.json` 会复用已完成候选，若要重做全部验证调参，仅在独立重跑目录移走该文件。每轮数据资格与排除按固定规则执行，不能通过改门槛“刷出”更好的成绩。

原报告中的 CPU 耗时是一次本机 Python 计算路径测量，排除传感器采样、传输、策略和执行器；新机器重跑得到不同耗时是正常的。训练种子误差条也不是硬件噪声或物理重复性。

## 数据引用与版本

见 [数据来源与使用边界](DATA_SOURCES.md)。发布数据的许可不由本仓库重新指定；HTT 原始数据不随本提交分发。不能把 TaF 的保存单位下力矩／力比称作已标定 CoP，或把其约 30 Hz 数据外推为 FlexiTac 的 100 Hz 实机表现。

## 归档校验

`archive_manifest.json` 保存本次发布文件的 SHA-256、体积、原结果路径、未上传的大型文件清单与原研究基准提交。`outputs/spatial_validation_expanded/reproducibility_manifest.json` 是原实验环境与脚本指纹；报告脚本仅对归档表述做了修改，因此当前发布版本指纹以 archive_manifest.json 为准。
