# 评估集 V1：Ozimops petersi

## 目的

这个受控评估集用于测试：单智能体工作流能否为单个蝙蝠回声定位叫声生成可靠的强标注。它提供长度较短、结构统一的音频片段，以及叫声级别的时频真值标注，用于建立仅基于声谱图的基线。

## 源数据集

本评估集来源于 BatDetect2 Australia 数据集。传递给 `build_evaluation_set.py` 的源数据目录必须包含：

```text
annotations.json
audio/*.wav
```

主要目标物种为 *Ozimops petersi*。选用的源录音为 `pseudo_petersi_001.wav` 至 `pseudo_petersi_010.wav`，共 10 个源录音文件。

## 标注标准

每个独立的回声定位叫声使用一个紧贴目标的时频边界框表示：

```text
[start_time, low_frequency, end_time, high_frequency]
```

边界框应当：

- 只包含一个独立叫声；
- 避开回声、混响和伪影；
- 只覆盖主要谐波；
- 对于确实属于回声定位叫声的低信噪比信号，只要在声谱图中可见，也应进行标注。

在未来的评估集中，如果物种无法确定，可以使用通用的 `Bat` 标签表示。本评估集中的标签均为 *Ozimops petersi*。

## 生成的评估集

生成目录的结构如下：

```text
ozimops_petersi_v1/
├── audio/
├── ground_truth/
└── manifest.csv
```

数据规模如下：

| 项目 | 数量 |
| --- | ---: |
| 源录音 | 10 |
| 一秒音频片段 | 45 |
| 唯一源事件 | 191 |
| 片段级事件实例 | 198 |

片段级事件实例数大于唯一源事件数，是因为有 7 个源事件跨越了一秒片段的边界。每个跨边界事件都会被纳入两个与其重叠的片段级真值文件中，同时保留原始 UUID，并将片段内的局部时间范围裁剪到对应片段的边界。

自动分配的场景分布如下：

| 场景 | 片段数 |
| --- | ---: |
| `multi_event` | 18 |
| `positive` | 27 |

## 边界截断

每个片段级事件都会记录切片过程是否改变了其可见时间范围：

- `is_truncated_by_clip_boundary`：当片段的任意一侧边界截断源事件时为 `true`。
- `truncation_side`：取值为 `none`、`left`、`right` 或 `both`。

`left` 表示源事件在当前片段开始之前已经开始；`right` 表示源事件在当前片段结束之后仍未结束。当前分布如下：

| 截断方向 | 事件实例数 |
| --- | ---: |
| `none` | 184 |
| `left` | 7 |
| `right` | 7 |
| `both` | 0 |

所有写入文件的时间值均保留至小数点后 6 位。

## Manifest 字段

| 字段 | 含义 |
| --- | --- |
| `clip_id` | 稳定的顺序标识符，例如 `OP_001`。 |
| `clip_path` | 生成 WAV 文件的相对路径。 |
| `ground_truth_path` | 片段级真值 JSON 文件的相对路径。 |
| `source_recording` | BatDetect2 Australia 数据集中的原始录音文件名。 |
| `source_start_time` | 当前片段在原始录音时间轴上的开始时间。 |
| `source_end_time` | 当前片段在原始录音时间轴上的结束时间。 |
| `clip_duration` | 片段的实际持续时间，包括最后一个不足完整时长的片段。 |
| `species` | 本评估集的目标物种。 |
| `has_target_event` | 当前片段是否包含至少一个目标事件。 |
| `num_gt_events` | 当前片段包含的片段级真值事件实例数。 |
| `event_density` | 事件密度：`zero`、`low`、`medium` 或 `high`。 |
| `auto_scenario` | 根据事件数量自动生成的活动场景。 |
| `manual_scenario` | 为后续人工声谱图检查预留的场景字段。 |
| `notes` | 用于后续数据整理的自由文本备注。 |

## 可复现构建命令

```bash
uv run python build_evaluation_set.py \
  --dataset-dir "<DATASET_ROOT>" \
  --output-dir outputs/evaluation_sets/ozimops_petersi_v1 \
  --species "Ozimops petersi" \
  --source-prefix "pseudo_petersi_" \
  --clip-seconds 1.0 \
  --overwrite
```

源数据集保留在项目仓库之外。生成的 WAV 文件和评估输出不会被 Git 跟踪。

## 验证摘要

```text
片段数：45
唯一源事件数：191
片段级事件实例数：198
所有路径均为相对路径：true
验证错误：无
```

## 局限性

- 这是一个只包含正样本的单物种评估集。
- 当前尚未包含负样本片段。
- 当前尚未测试多物种泛化能力。
- `clear_call`、`weak_call`、`noisy` 或 `borderline` 等人工场景标签尚未分配。后续可在抽查声谱图后补充这些标签。

## 后续工作

1. 抽查带有真值边界框的生成声谱图。
2. 为 prompt v2 选择具有代表性的标注示例。
3. 使用该评估集开展仅基于声谱图的单智能体基线评估。
