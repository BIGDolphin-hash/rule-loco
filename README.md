# Rule-LOCO

面向工业视觉异常检测的显式规则与正常记忆库研究实现。本次源码更新包含五类规则、共享 SAM3 掩码、DINOv2 特征提取，以及 GPO-JM（Global Patch Ordered Joint Memory）。

## 核心流程

1. 从通用规则模板收集对象类别，使用 SAM3 分割并进行同类别掩码去重。
2. 使用共享掩码计算 COUNT、LENGTH、AREA、SPATIAL_COMBINATION 和 ATTRIBUTE_COMBINATION 五类规则证据。
3. `binary-selective` 根据规则结果决定是否进入记忆分支；规则未全部通过时提前退出。`binary-parallel` 提供并行执行路径，具体评分与异常状态处理见 `pipeline.py`。
4. GPO-JM 按固定类别顺序生成语义描述和纹理统计，通过联合距离匹配正常记忆，输出异常分数。CLIP 仅用于可选属性判断。

## 主要代码

| 文件 | 作用 |
|---|---|
| `src/pro_innovation/pipeline.py` | 规则推理、分支调度和最终评分 |
| `src/pro_innovation/memory_modules/gpojm/core.py` | GPO-JM 联合记忆、距离、阈值和评分 |
| `src/pro_innovation/memory_modules/gpojm/semantic.py` | 类别有序语义描述 |
| `src/pro_innovation/memory_modules/gpojm/global_patch.py` | GPO-JM 专用正常 Patch 记忆与匹配 |
| `src/pro_innovation/adapters/sam3.py` | SAM3 分割适配 |
| `src/pro_innovation/adapters/patch_dinov2.py` | DINOv2 多层特征提取 |
| `src/pro_innovation/mask_processing.py` | 共享掩码后处理 |
| `src/pro_innovation/templates.py` | 模板解析与校验 |
| `src/pro_innovation/geometry.py`, `spatial.py`, `attributes.py` | 几何、空间和属性证据 |
| `src/pro_innovation/models.py` | 规则、掩码和报告数据结构 |
| `scripts/global_patch_ordered/build_memory.py` | 构建 GPO-JM 联合记忆 |
| `scripts/global_patch_ordered/evaluate_rule_gpojm_mvtec_loco.py` | 五类规则与 GPO-JM 评估入口 |

## 安装与外部依赖

Python >= 3.10。按本机 CUDA 环境安装 PyTorch，然后安装此包：

```bash
python -m pip install -e .
```

运行完整模型还需要自行安装 SAM3、DINOv2，以及使用属性模型时的 CLIP 官方代码及其依赖，并下载相应模型权重。本仓库不打包这些第三方代码或权重。路径默认采用 `models/`、`weights/` 等相对位置，也可通过命令行参数指定。

```bash
PYTHONPATH=src python -m pro_innovation --help
PYTHONPATH=src python scripts/global_patch_ordered/build_memory.py --help
PYTHONPATH=src python scripts/global_patch_ordered/evaluate_rule_gpojm_mvtec_loco.py --help
```

## 建库与评估

`examples/` 提供模板格式示例。请自行准备目标类别的通用与标准模板，以及合法取得的数据集。`config/` 中包含属性词表及五类对象的 SAM3 阈值配置；阈值配置中的标定图像路径仅是来源记录，不包含图像。

GPO-JM 联合建库需要预先准备兼容的底层 Patch 记忆文件（`--global-bank`）。当前发布版只提供 GPO-JM 联合建库和推理代码，不包含该底层记忆文件的生成器；记忆文件需要在外部准备。联合建库示例：

```bash
PYTHONPATH=src python scripts/global_patch_ordered/build_memory.py \
  --class-name screw_bag \
  --normal-dir datasets/mvtec_loco_anomaly_detection/screw_bag/train/good \
  --generic templates/store/screw_bag.generic.rules \
  --threshold-policy config/sam3_thresholds/confidence_gap_midpoint_4-shot/screw_bag.json \
  --global-bank memory_banks/screw_bag.global.npz \
  --output memory_banks/screw_bag.joint.npz
```

评估参数及可用调度选项以入口的 `--help` 为准。数据、权重、模板实例和记忆库需要在本地准备；上述命令不表示仓库包含这些文件。

## 发布范围

本仓库只发布源码、必要配置和模板格式示例。不发布数据集、模型权重、正常记忆库、缓存、实验结果、运行日志、本地工具状态、论文草稿或本地模板实例。
