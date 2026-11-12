# 配置目录索引

按用途保存配置；选样清单、模型配置和任务定义分别维护。

源码发布包含属性词表、UTS、阈值策略和推理依赖。`vlm/`、`data_selection/` 及完整环境快照仅在本机维护，因此克隆仓库后可能不存在这些目录或文件。

| 目录 | 内容 |
| --- | --- |
| [attributes/](attributes/) | CLIP/规则推理使用的属性候选词表，按类别命名 |
| [template_generation/](template_generation/) | 模板生成使用的 UTS 通用任务定义 |
| `vlm/` | 本机 VLM 服务、模型、地址和凭据配置 |
| `data_selection/` | Patch 建库和 few-shot 实验的正常图像选样清单 |
| [sam3_thresholds/](sam3_thresholds/) | SAM3 对象分割阈值策略，按标定方法组织 |
| [environment/](environment/) | 环境定义、依赖、模型来源和本机运行说明 |

## 常用入口

- 属性词表示例：[attributes/example.json](attributes/example.json)。类别词表为 `attributes/<class_name>.json`，现有 `juice_bottle` 和 `splicing_connectors` 两份。
- UTS 任务定义：[template_generation/uts.v1.json](template_generation/uts.v1.json)。
- 本机 VLM 配置：`vlm/vlm.local.json`；运行时默认自动读取，可用 `VLM_CONFIG_FILE` 显式指定其他文件。该目录不进入源码发布，本机配置保持 `0600` 权限。
- 完整 Patch 选样：`data_selection/patch/mvtec_ad.json`、`data_selection/patch/visa.json`（本机）。
- Few-shot 清单：`data_selection/few_shot/mvtec_ad/`、`data_selection/few_shot/visa/`（本机）。同一数据集的随机选样和分层选样清单分别保留，文件名标明 seed、shot 或选样方法。
- 当前阈值策略：[sam3_thresholds/confidence_gap_midpoint_4-shot/](sam3_thresholds/confidence_gap_midpoint_4-shot/)。
- 推理依赖：[environment/requirements-inference.txt](environment/requirements-inference.txt)；完整本机运行说明为 `environment/README.md`。

## 路径调整

2026-11-12 将散落文件归入以上目录。仅移动配置，原有内容与文件权限保持不变；代码默认值、Shell 脚本、测试、文档、Git 发布规则及已有建库日志中的相关路径同步更新。`sam3_thresholds/` 与 `environment/` 保持原路径。

| 原路径（相对 config/） | 当前路径（相对 config/） |
| --- | --- |
| `attribute_vocabulary.example.json` | `attributes/example.json` |
| `juice_bottle.attribute_vocabulary.json` | `attributes/juice_bottle.json` |
| `splicing_connectors.attribute_vocabulary.json` | `attributes/splicing_connectors.json` |
| `uts.v1.json` | `template_generation/uts.v1.json` |
| `vlm.local.json` | `vlm/vlm.local.json` |
| `mvtec_ad_patch_selection.json` | `data_selection/patch/mvtec_ad.json` |
| `visa_patch_selection.json` | `data_selection/patch/visa.json` |
| `few_shot/mvtec_ad/seed0_shots-1-2.json` | `data_selection/few_shot/mvtec_ad/seed0_shots-1-2.json` |
| `few_shot/visa/seed0_shots-1-2.json` | `data_selection/few_shot/visa/seed0_shots-1-2.json` |
| `mvtec_ad_few_shot/seed0_shots-2-4.json` | `data_selection/few_shot/mvtec_ad/seed0_shots-2-4.json` |
| `visa_castjm_4shot/stratum_center_4shot.json` | `data_selection/few_shot/visa/stratum_center_4shot.json` |
