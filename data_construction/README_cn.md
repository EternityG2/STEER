# STEER 数据构建说明

本目录包含 **STEER** 的数据构建流程，用于构造两类数据：一类是训练 Test Generator 的任务级数据，另一类是训练 Process Reward Model（PRM）的步骤级监督数据。

整体流程从 **CodeContests+** 出发，生成测试用例轨迹，通过 Monte Carlo rollout 估计步骤质量，再用 LLM Judge 进行语义验证，最后将共识样本转换成 OpenRLHF 可用的 PRM 数据格式。

## 目录结构

```text
data_construction/
├── data_init_process.py                   # 从 CodeContests+ 初始化任务
├── filter_by_token.py                     # 按 token 长度过滤任务
├── filter_difficulty_by_llm.py            # 使用 LLM 保留中等难度任务
├── build_balanced_dataset.py              # 构造平衡的正/负轨迹
├── compute_mc_scores_ray.py               # 通过 MC rollout 估计步骤质量
├── convert_mc_scores_to_hard_labels.py    # 将 MC 分数转换为硬标签
├── llm_as_judge.py                        # 使用 LLM Judge 验证步骤
├── strict_consensus_filter_mc_judge.py    # 保留 MC/Judge 一致的标签
├── convert_consensus_to_openrlhf_prm.py   # 转换为 OpenRLHF PRM 数据
├── prompts.py                             # Prompt 模板
└── run_data_construction.sh               # 端到端启动脚本
```



## 快速开始

请先启动生成模型和 Judge 模型服务。接口需要兼容 OpenAI API，例如使用 vLLM 启动本地服务。

```bash
cd data_construction

export TOKENIZER_MODEL_PATH=

export GENERATION_API_BASE_URL=
export GENERATION_API_KEY=
export GENERATION_MODEL_NAME=

export JUDGE_API_BASE_URL=
export JUDGE_API_KEY=
export JUDGE_MODEL_NAME=

bash run_data_construction.sh
```

默认输出目录为：

```text
outputs/data_construction/
```

也可以通过 `OUT_DIR` 指定输出位置：

```bash
export OUT_DIR=/path/to/output/data_construction
bash run_data_construction.sh
```

## 必需环境变量

| 变量                      | 说明                                                     |
| ------------------------- | -------------------------------------------------------- |
| `TOKENIZER_MODEL_PATH`    | 本地 tokenizer/model 路径，用于长度过滤和 PRM 数据转换。 |
| `GENERATION_API_BASE_URL` | 生成模型的 OpenAI-compatible API 地址。                  |
| `GENERATION_API_KEY`      | 生成模型 API key。本地服务不需要 key 时可设为 `EMPTY`。  |
| `GENERATION_MODEL_NAME`   | 生成模型服务名称。                                       |
| `JUDGE_API_BASE_URL`      | Judge 模型的 OpenAI-compatible API 地址。                |
| `JUDGE_API_KEY`           | Judge 模型 API key。                                     |
| `JUDGE_MODEL_NAME`        | Judge 模型服务名称。                                     |

## 常用可选环境变量

| 变量                     |                              默认值 | 说明                                                         |
| ------------------------ | ----------------------------------: | ------------------------------------------------------------ |
| `DATASET_NAME`           | `ByteDance-Seed/Code-Contests-Plus` | 源数据集名称。                                               |
| `DATASET_CONFIG`         |                                `3x` | 数据集配置。                                                 |
| `DATASET_SPLIT`          |                             `train` | 数据集划分。                                                 |
| `TARGET_DIFFICULTIES`    |                               `7,8` | 初始化阶段保留的难度等级。                                   |
| `MAX_TOKENS`             |                              `2000` | token 长度过滤阈值。                                         |
| `GENERATION_TEMPERATURE` |                               `0.8` | 生成和 MC rollout 的采样温度。                               |
| `GENERATION_MAX_TOKENS`  |                              `2048` | 最大生成长度。                                               |
| `EXEC_TIMEOUT`           |                                 `5` | 执行生成测试用例的超时时间。                                 |
| `RAY_NUM_ACTORS`         |                                 `8` | 生成/MC 阶段的 Ray actor 数量。                              |
| `MAX_SAMPLES_PER_TASK`   |                                `16` | MC/Judge 阶段每个任务最多使用的样本数。                      |
| `MAX_MC_STEPS`           |                                 `3` | MC 阶段最多评估的推理步骤数。                                |
| `RPE_EPSILON`            |                               `0.8` | MC 分数转硬标签的相对进展阈值。复现实验主设置时可设为 `0.9`。 |
| `JUDGE_TEMPERATURE`      |                               `0.2` | Judge 模型采样温度。                                         |
| `JUDGE_RAY_NUM_ACTORS`   |                                 `8` | Judge 阶段的 Ray actor 数量。                                |
| `TEST_ROWS`              |                               `500` | OpenRLHF 转换时测试集样本数。                                |
| `SEED`                   |                                `42` | 随机种子。                                                   |

## 输出文件

```text
outputs/data_construction/
├── init/
│   ├── raw_filtered.jsonl
│   ├── token_filtered.jsonl
│   └── tasks.jsonl
├── balanced/
│   └── balanced_tasks.jsonl
├── mc_scores/
│   ├── mc_tasks.jsonl
│   ├── mc_hard_labels.jsonl
│   └── hard_label_stats.json
├── judge/
│   └── judge_tasks.jsonl
├── consensus/
│   ├── retained.jsonl
│   ├── all_with_flags.jsonl
│   └── stats.json
└── openrlhf/
    └── ...
```