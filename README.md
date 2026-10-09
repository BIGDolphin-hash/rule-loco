# Pro Innovation V1

这是一个“显式规则 + DINOv2 Patch Matching”的图像级异常检测实现。
规则分支负责数量、多余/缺失对象、长度、面积和模板关系；Patch 分支负责
局部结构异常。两条分支最终融合成一个 `image_anomaly_score`。

```text
类别通用模板（无 VALUE）
→ 只收集 SUBJECT / OBJECT / SUBJECT_n
→ 类别去重
→ SAM3 对每个类别分割一次
→ 每个 Mask 保留最大连通主体，并在同类别内抑制高度重叠的重复 Mask
→ 建立共享 Mask 池
→ 六类任务分别计算实际值
→ 填充测试实例模板
→ 与标准模板逐规则比较并计算约束满足分数
→ 统一转换为 violation_score = 1 - satisfaction_score
→ 可选DINOv2 Patch Matching（默认关闭）
→ 规则与 Patch 分支以 0.5 为统一边界后取最大证据
→ 一个整图 image_anomaly_score
```

## DINOv2 Patch Matching（LogSAD few-shot DINO路径）

固定使用本机 DINOv2 ViT-L/14-Reg4、`448×448` 输入和 Block 6/12/18/24。
四层 `32×32` Patch 特征均双线性插值到 `64×64` 并做 L2 归一化。每层查询
Patch 与四张正常记忆图的该层全部 Patch 做全局余弦最近邻匹配，不使用位置
半径、Coreset、Top-K、逐层阈值或单层消融：

```text
A_l(i) = 1 - max_j cosine(q_l,i, m_l,j)
A(i)   = (A_6(i) + A_12(i) + A_18(i) + A_24(i)) / 4
s_raw  = max_i A(i)
s_std  = (s_raw - mean_validation) / unbiased_std_validation
s_patch = sigmoid(s_std)
```

四张指定正常图只负责记忆库；独立的 `validation/good` 正常图负责计算最终
结构分数的均值和无偏标准差。`s_patch > 0.5` 判为 Patch 异常。Patch 分支仅
处理完整图像，不使用 SAM3 Mask 或目标裁剪。

建立记忆库：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation build-patch-memory \
  --class-name <class-name> \
  --normal-images <normal-1.png> <normal-2.png> <normal-3.png> <normal-4.png> \
  --calibration-dir <dataset>/<class-name>/validation/good \
  --device cuda
```

默认保存到：

```text
/home/lxq/pro-innovation/memory_banks/4-shot/mvtec_loco/<class-name>.dino_b6b12b18b24.npz
```

记忆库按样本数和数据集分层：

```text
memory_banks/
└── 4-shot/
    ├── mvtec_loco/
    ├── mvtec_ad/
    └── visa/
```

`evaluate --dataset ...` 会自动到对应的数据集目录查找同名四样本 bank；显式
传入 `--patch-bank` 时仍以该路径为准。

旧 Patch 库 schema 不兼容，加载时会明确要求重建。Patch Matching 默认关闭，
使用 `--patch-matching` 开启；完整测试集仍需 `--all-test-splits`：

```bash
python -m pro_innovation evaluate --class-name screw_bag \
  --patch-matching --all-test-splits
```

规则路线默认开启。只运行完整图像 Patch 分支时，使用
`--no-rule-inference --patch-matching`。此模式不读取规则模板或
SAM3 阈值策略，也不加载 SAM3、规则属性 CLIP 和 Mask 后处理模块：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation evaluate \
  --class-name <class-name> \
  --no-rule-inference \
  --patch-matching \
  --patch-bank <patch-memory.npz> \
  --test-splits good logical_anomalies \
  --restart
```

Patch-only 结果文件名包含 `patchonly`，摘要中的 `execution_mode` 为
`patch_only`，且 `logical_anomaly`、模板和 SAM3 配置均为空。切换该运行开关
不会改变 Patch 特征或记忆库 schema，因此不需要重建 Patch memory bank。

每条 JSONL 的 `patch_evidence` 记录四层各自的最大 Patch 距离（仅供诊断）、
四层异常图平均后的 `aggregate_raw_score`、正常验证统计量以及 sigmoid 后的
`calibrated_score`。最终模型只使用这一个 `calibrated_score`。

## 已实现

- 六类固定规则的严格解析、校验和序列化。
- 标准模板自动派生无值的类别通用模板。
- 同一图像只计算一次 SAM3 图像特征，每个唯一类别只发送一次文本提示词。
- SAM3 输出先做 Mask 清理和同类别空间去重；不同位置的同类物体不会因类别或长度相同而合并。
- COUNT、相对 LENGTH、相对 AREA、空间关系、单属性、属性组合六条执行路径。
- LENGTH 采用 Mask 前景点主轴投影长度；AREA 采用前景像素数。
- 长度/面积严格按“测量 → 相似值分组 → 降序排名 → 组内计数”执行。
- 简单颜色和粗粒度形状使用 NumPy/Pillow；复杂语义属性通过本地 CLIP 候选集分类。
- 测试实例模板回填，以及逐规则 PASS/FAIL、满足分数、违反分数和最终整图异常报告。
- 所有数量真值都使用非空整数数组：COUNT 任务写作 `VALUE=[1]` 或 `VALUE=[1,2]`，LENGTH/AREA 以及可选的空间关系数量写作 `COUNT=[1]` 或 `COUNT=[1,2]`；实际数量命中数组中的任一值即通过。
- COUNT 使用 SAM3 Mask 置信度计算数量落入允许数组的支持度。
- LENGTH、AREA 对 Mask 存在性进行子集边缘化，再执行相对等级和允许数量数组判断。
- SPATIAL_COMBINATION 从标准模板的 `VALUE` 读取关系名，并根据连续几何证据生成关系支持度；支持方向、包含、距离以及 `middle`、`lower`、`center`。可选的 `COUNT=[...]` 统计至少有一个 OBJECT 满足该关系的唯一 SUBJECT 数量。
- ATTRIBUTE_ERROR 使用正确属性的候选概率；ATTRIBUTE_COMBINATION 可在 `VALUE` 中保存允许组合列表，组合内部取最弱成员、组合之间取最大支持度。
- 未启用 Patch 时，规则分支取有效约束 `violation_score` 的最大值。启用
  Patch 分支后，
  用规则的 PASS/FAIL 判定校准尺度：PASS 分数映射到 `[0,0.5)`，FAIL 分数
  映射到 `(0.5,1]`；再与以正常阈值对应 `0.5` 的 Patch 证据取最大值。融合结果
  是统一判定边界上的排序证据，不解释为概率。
- 开启 LogSAD DINO Patch 路径后的评估文件名包含
  `dino_b6b12b18b24`。该路径固定使用官方四层异常图平均方式，不再提供
  单层、mean/max 或位置半径消融。各分支在统一的 `0.5`
  判定边界上直接比较，最高分来源决定异常类型；
  精确并列时输出 `MIXED_ANOMALY`。
- 本地SAM3、DINOv2、CLIP的惰性加载适配器，不会自动下载权重。

空间关系未写 `COUNT` 时采用“任意一对实例满足 `VALUE` 指定关系即通过”；写入 `COUNT=[...]` 后，则逐个 SUBJECT 寻找至少一个满足关系的 OBJECT，并对满足关系的唯一 SUBJECT 数量进行判断。同一 SUBJECT 即使匹配多个 OBJECT 也只计数一次。该计数适用于 `left_of`、`right_of`、`above`、`below`、`overlap`、`disjoint`、`touching`、`inside`、`contains`、`near`、`far`、`middle`、`lower`、`center`。`inside`/`contains` 与 `center` 一样使用外层掩码的边界框作为填充的对象相对空间，并计算内层掩码像素位于该空间内的比例；默认至少 `0.95` 才成立，不要求两个语义掩码本身重叠。其中 `middle` 使用参照目标掩码的主方向建立局部坐标系，再在归一化坐标中判断二维中部区域，因此不依赖类别、图像尺度或目标旋转角度。

## 当前本地模型配置

- SAM3 仓库：`/home/lxq/models/SAM3`
- SAM3 权重：`/home/lxq/weights/sam3/sam3.pt`
- CLIP 仓库：`/home/lxq/models/CLIP`
- CLIP 权重：`/home/lxq/weights/CLIP/ViT-L-14.pt`
- DINOv2 Patch权重：`/home/lxq/weights/DINOV2/dinov2_vitl14_reg4_pretrain.pth`
  （ViT-L/14-Reg4；仅使用本机文件，不会自动下载）
- 推荐统一使用现有 `sam3` Conda环境。

代码只在建立Patch/外观记忆或实际推理/评估时惰性加载对应模型。模板处理和
单元测试不会占用GPU。

## 模板操作

在项目根目录执行：

```bash
PYTHONPATH=src conda run -n sam3 python -m pro_innovation validate examples/standard.rules
PYTHONPATH=src conda run -n sam3 python -m pro_innovation validate --generic examples/generic.rules
PYTHONPATH=src conda run -n sam3 python -m pro_innovation make-generic examples/standard.rules
```

把多模态大模型生成并经人工确认的标准模板保存为标准/通用模板对：

```bash
PYTHONPATH=src conda run -n sam3 python -m pro_innovation save-standard demo examples/standard.rules
```

多模态大模型的具体供应方和调用接口尚未指定，因此 V1 定义了生成器接口，并从“接收生成结果、严格校验、保存”开始实现，未绑定或调用任何付费 API。

## 本地推理入口

复杂语义属性需要候选词表，示例见 `config/attribute_vocabulary.example.json`：

候选词表字段必须使用 `{SUBJECT}_{属性名}`，与模板中的属性主体严格绑定，
不再接受全局 `type`、`color` 等字段。例如 `SUBJECT=juice_bottle` 与
`PROPERTY=attribute.color` 对应 `juice_bottle_color`；当前果汁瓶模板使用
`fruit_icon_type` 和 `liquid_color`。这样不同主体的同名属性不会共用或串用
CLIP 候选集。

```bash
PYTHONPATH=src conda run -n sam3 python -m pro_innovation infer \
  --class-name <class-name> \
  --image /path/to/test.png \
  --generic examples/generic.rules \
  --standard examples/standard.rules \
  --attribute-vocabulary config/attribute_vocabulary.example.json \
  --output report.json
```

退出码：正常为 `0`，检测到逻辑/结构/混合异常为 `1`，规则证据不足、模板、
配置或运行错误为 `2`。仅记忆分支本身不存在 `UNKNOWN`。

单条规则报告统一包含：

```text
satisfied
satisfaction_score
violation_score
evidence_valid
actual
expected
details
```

如果某条规则无法执行，其 `satisfied` 和两个分数均为 `null`，并标记
`evidence_valid=false`，不会把模型或证据失败伪装成产品异常。

## 测试

```bash
PYTHONPATH=src conda run -n sam3 python -m unittest discover -s tests -v
```

测试使用合成 Mask 和模拟模型，不会运行 SAM3、CLIP 或下载文件。

## 四张正常真值图生成对象级 SAM3 阈值

在测试集推理前，可用4张互不相同的正常图为每个对象生成固定阈值。校准阶段
SAM3 的候选生成阈值默认是 `0.10`，也可以通过 `--sam3-threshold` 动态指定
`[0,1]` 范围内的值，随后按每张图具体模板中的真实数量 `K`
读取第 `K` 高分数和第 `K+1` 高分数。跨4张图计算额外候选的Q95与必需候选
的Q05，二者存在间隔时先计算中点，再向下取到 `0.10` 档位；例如
`0.719845` 保存为 `0.70`，最终阈值只按 `0.10` 档位向下取整，不额外设置固定最低值。整个过程只统计推理结果，
`model_parameters_updated=false`，不会更新SAM3参数。

当4张图使用相同且每个COUNT均为单值的标准模板时：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation calibrate-sam3-thresholds \
  --class-name <class-name> \
  --normal-images <normal-1.png> <normal-2.png> <normal-3.png> <normal-4.png>
```

如果共享模板写的是 `VALUE=[4,6,10]` 这类允许集合，必须提供与4张图逐一对应、
COUNT已经具体化为单值的真值模板：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation calibrate-sam3-thresholds \
  --class-name <class-name> \
  --normal-images <normal-1.png> <normal-2.png> <normal-3.png> <normal-4.png> \
  --truth-templates <truth-1.rules> <truth-2.rules> <truth-3.rules> <truth-4.rules>
```

默认保存到
`/home/lxq/pro-innovation/config/sam3_thresholds/<class-name>.json`。
`infer/evaluate/evaluate-screw-bag` 会按类别自动加载；未成功校准的对象继续使用
全局 `--sam3-threshold`。策略会校验SAM3提示词、权重路径、标准模板内容及Mask
后处理配置，防止加载不匹配的旧阈值。消融时使用
`--no-object-sam3-thresholds`，代码和保存结果均保留。

## 通用测试集评估

`evaluate` 会根据 `--class-name` 自动选择：

```text
/home/lxq/pro-innovation/datasets/mvtec_loco_anomaly_detection/<class-name>/test
/home/lxq/pro-innovation/templates/store/<class-name>.generic.rules
/home/lxq/pro-innovation/templates/store/<class-name>.standard.rules
```

Patch 默认关闭；默认读取 `good + logical_anomalies`。无论开启哪个
模型模块，要评估完整测试集都需添加 `--all-test-splits`，此时读取
`good + logical_anomalies + structural_anomalies`。模型开关不会暗中改变评估范围。
含复杂语义属性的类别需要显式提供候选词表。
例如：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation evaluate \
  --class-name juice_bottle \
  --attribute-vocabulary config/attribute_vocabulary.example.json \
  --device cuda \
  --sam3-threshold 0.40
```

其他数据位置或模板位置可分别通过 `--dataset-root`、`--generic` 和
`--standard` 覆盖。输出文件名使用实际类别、模型阈值、Mask 处理配置及
模板中的 LENGTH/AREA 容差自动生成。中断后重复同一命令会续跑；使用
`--restart` 才会重新开始。

CLIP 不再对 SAM3 Mask 做类别复核或硬过滤；模板含复杂语义属性规则时，
CLIP 只用于属性候选分类，`--clip-threshold`（默认 `0.35`）是属性预测的最低
置信度；低于该值时对应规则输出 `UNKNOWN`。该参数不会重新开启 SAM3 Mask
类别复核。

`attribute.color` 使用确定性的平均 RGB 最近色，并且只会在属性词表的
`color` 列表中选择；缺少该列表或配置了不支持的颜色时会在模型加载前报错。

### MVTec AD 数据入口

项目内的数据软链接如下：

```text
datasets/mvtec_anomaly_detection
→ /home/lxq/Data/mvtec_anomaly_detection
```

使用 `--dataset mvtec_ad` 选择经典 MVTec AD 布局。代码会自动读取
`<class-name>/test/good`，并把同级的所有其他目录识别为正样本缺陷类型；也可
用 `--test-split` 或 `--test-splits` 只选择指定目录。MVTec AD 当前没有对应的
规则模板，因此推荐使用完整图像 Patch-only 路径：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation evaluate \
  --dataset mvtec_ad \
  --class-name bottle \
  --no-rule-inference \
  --patch-matching \
  --patch-bank memory_banks/4-shot/mvtec_ad/bottle.dino_b6b12b18b24.npz \
  --device cuda
```

MVTec AD 的每个类别都需要用本机 ViT-L/14-Reg4 权重单独建立 Patch memory
bank；不能复用 MVTec LOCO 类别的旧库。`--dataset-base` 和 `--dataset-root`
仍可覆盖默认软链接入口。

一次建立或续建全部 15 类 Patch bank：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python scripts/build_mvtec_ad_patch_banks.py
```

脚本按排序文件名的四个分层中心选择四张记忆图，并从其余训练正常图中另选
16 张做独立校准；选择清单保存在
`config/mvtec_ad_patch_selection.json`。再次执行会加载并验证已有 bank，只续建
缺失类别；间歇性子进程失败最多自动重试三次。

### VisA 数据入口

项目内的数据软链接如下：

```text
datasets/visa
→ /home/lxq/Data/VisA
```

使用 `--dataset visa` 选择 VisA。入口读取官方 `split_csv/1cls.csv`，只评测
指定类别的 `test` 行，并将 `Normal` 作为负样本、`Anomaly` 作为正样本；不会
把训练正常图混入评测。VisA 图像是 `.JPG`，路径直接取自官方清单。例如：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python -m pro_innovation evaluate \
  --dataset visa \
  --class-name candle \
  --no-rule-inference \
  --patch-matching \
  --patch-bank memory_banks/4-shot/visa/candle.dino_b6b12b18b24.npz \
  --device cuda
```

每个 VisA 类别需要独立建立与校准 Patch memory bank，不能复用其他数据集或
其他类别的 bank。`--dataset-base` 和 `--dataset-root` 对 VisA 都应指向包含
`split_csv` 的 VisA 根目录。

一次建立或续建全部 12 类 Patch bank：

```bash
PYTHONPATH=src conda run --no-capture-output -n sam3 \
  python scripts/build_visa_patch_banks.py
```

脚本只从官方 `train + normal` 行取样：按排序路径的四个分层中心选择 4 张
记忆图，再从剩余训练正常图选择 16 张独立校准图。选择清单保存在
`config/visa_patch_selection.json`；已有 bank 通过元数据校验后会跳过，因此
中断后可直接执行同一命令续建。DINOv2 默认使用 CUDA BF16；若某批中间特征
出现非有限值，会仅对该批自动回退到 FP32 重算。

## screw_bag 兼容入口

阈值直接通过命令传入，结果文件名会自动包含 SAM3 阈值和标准模板中的
长度容差。例如：

```bash
PYTHONPATH=/home/lxq/pro-innovation/src conda run --no-capture-output -n sam3 \
  python -m pro_innovation evaluate-screw-bag \
  --sam3-threshold 0.40 \
  --mask-dedup \
  --mask-cleanup \
  --mask-dedup-iou 0.80 \
  --mask-dedup-containment 0.85 \
  --mask-min-area 16
```

当前 ±5% 长度模板会自动生成：

```text
/home/lxq/pro-innovation/result/screw_bag_sam040_dedup080contain085_minarea16_length5.jsonl
/home/lxq/pro-innovation/result/screw_bag_sam040_dedup080contain085_minarea16_length5_summary.json
```

同参数中断后重跑会自动续跑；显式添加 `--restart` 才会从头运行。

Mask 去重和清理默认开启。消融时可分别使用 `--no-mask-dedup` 和
`--no-mask-cleanup`；数值参数及其实现均保留。关闭清理会同时跳过最大连通域
清理和 `--mask-min-area` 过滤。去重在同类别内同时检查 IoU 和“小掩码被包含率”；
任一指标达到 `--mask-dedup-iou` 或 `--mask-dedup-containment` 阈值，就抑制
低置信度候选。关闭去重会跳过这两个判断。

## 第一版边界

- CLIP 必须使用配置中的候选属性值，标准模板的正确答案不会作为推理候选动态注入，避免答案泄漏。
- 简单形状目前仅输出 `square`、`round`、`elongated`，属于可替换的粗粒度规则。
- 空间关系目前是几何启发式；后续可按具体数据集定义配对策略和阈值。
- 当前未实现某一多模态大模型的在线标准模板生成，因为尚未确定本地模型或 API。
- 当前连续值是统一量纲的证据分数，还没有使用带标签验证集做概率校准，因此正式名称为 `satisfaction_score` / `violation_score`，不能直接宣称为真实概率。
- Mask 分数只能描述 SAM3 已提出候选的可信程度，不能估计 SAM3 完全漏掉目标的概率。
- 本轮只完成轻量测试，不进行正式 SAM3/CLIP 推理或评估。
