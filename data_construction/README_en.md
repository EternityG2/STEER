# STEER Data Construction

This directory contains the data construction pipeline for **STEER**, which builds task-level data for the Test Generator and step-level supervision data for the Process Reward Model (PRM).

The pipeline starts from **CodeContests+**, generates bug-exposing test cases, estimates step quality with Monte Carlo rollouts, verifies labels with an LLM judge, and finally exports consensus labels to the OpenRLHF PRM format.

## Directory Structure

```text
data_construction/
├── data_init_process.py                   # Load initial tasks from CodeContests+
├── filter_by_token.py                     # Filter long tasks by tokenizer length
├── filter_difficulty_by_llm.py            # Keep moderately difficult tasks
├── build_balanced_dataset.py              # Build balanced positive/negative trajectories
├── compute_mc_scores_ray.py               # Estimate step quality with MC rollouts
├── convert_mc_scores_to_hard_labels.py    # Convert MC scores to hard labels
├── llm_as_judge.py                        # Verify steps with an LLM judge
├── strict_consensus_filter_mc_judge.py    # Keep MC/Judge-consistent labels
├── convert_consensus_to_openrlhf_prm.py   # Convert labels to OpenRLHF PRM data
├── prompts.py                             # Prompt templates
└── run_data_construction.sh               # End-to-end launcher
```

## Quick Start

Start the generation and judge model servers first. The APIs should follow the OpenAI-compatible format, for example through vLLM.

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

By default, all outputs are saved to:

```text
outputs/data_construction/
```

To use another output path:

```bash
export OUT_DIR=/path/to/output/data_construction
bash run_data_construction.sh
```

## Required Variables

| Variable                  | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `TOKENIZER_MODEL_PATH`    | Local tokenizer/model path used for token filtering and PRM conversion. |
| `GENERATION_API_BASE_URL` | OpenAI-compatible endpoint for the generation model.         |
| `GENERATION_API_KEY`      | API key for generation. Use `EMPTY` for local servers if no key is required. |
| `GENERATION_MODEL_NAME`   | Name of the served generation model.                         |
| `JUDGE_API_BASE_URL`      | OpenAI-compatible endpoint for the judge model.              |
| `JUDGE_API_KEY`           | API key for the judge model.                                 |
| `JUDGE_MODEL_NAME`        | Name of the served judge model.                              |

## Common Optional Variables

| Variable                 |                             Default | Description                                                  |
| ------------------------ | ----------------------------------: | ------------------------------------------------------------ |
| `DATASET_NAME`           | `ByteDance-Seed/Code-Contests-Plus` | Source dataset.                                              |
| `DATASET_CONFIG`         |                                `3x` | Dataset config.                                              |
| `DATASET_SPLIT`          |                             `train` | Dataset split.                                               |
| `TARGET_DIFFICULTIES`    |                               `7,8` | Difficulty levels kept at initialization.                    |
| `MAX_TOKENS`             |                              `2000` | Maximum token length for filtering.                          |
| `GENERATION_TEMPERATURE` |                               `0.8` | Sampling temperature for generation and MC rollouts.         |
| `GENERATION_MAX_TOKENS`  |                              `2048` | Maximum generation length.                                   |
| `EXEC_TIMEOUT`           |                                 `5` | Timeout for executing generated tests.                       |
| `RAY_NUM_ACTORS`         |                                 `8` | Number of Ray actors for generation/MC stages.               |
| `MAX_SAMPLES_PER_TASK`   |                                `16` | Maximum samples used per task in MC/Judge stages.            |
| `MAX_MC_STEPS`           |                                 `3` | Maximum reasoning steps evaluated by MC.                     |
| `RPE_EPSILON`            |                               `0.8` | Relative-progress threshold for hard labels. Use `0.9` to match the main paper setting. |
| `JUDGE_TEMPERATURE`      |                               `0.2` | Sampling temperature for the judge model.                    |
| `JUDGE_RAY_NUM_ACTORS`   |                                 `8` | Number of Ray actors for judging.                            |
| `TEST_ROWS`              |                               `500` | Test split size for OpenRLHF conversion.                     |
| `SEED`                   |                                `42` | Random seed.                                                 |

## Outputs

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

The most important final files are:

- `balanced/balanced_tasks.jsonl`: balanced positive/negative trajectories.
- `mc_scores/mc_hard_labels.jsonl`: hard step labels derived from MC scores.
- `judge/judge_tasks.jsonl`: judge decisions for step validity.
- `consensus/retained.jsonl`: retained labels where MC and judge agree.
- `openrlhf/`: PRM training/test data converted for OpenRLHF.

