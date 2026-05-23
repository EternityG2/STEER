# STEER: Enhancing Bug-Exposing Test Generation via Process-Supervised Training and Reward-Guided Search

[中文说明](./README_cn.md)

STEER is a process-supervised framework for bug-exposing test generation. Given a programming problem and an incorrect solution, the goal is to generate a test case that both exposes the bug and provides the correct expected output. STEER introduces a task-specific Process Reward Model (PRM) to improve this task in two places: it provides process rewards during reinforcement learning, and it guides test-time search over candidate reasoning paths.

The framework contains four main stages: dataset construction, PRM training, test generator training, and evaluation.

## Repository Structure

```text
STEER/
├── data_construction/   # Build PRM data and generator training data
├── datasets/            # Dataset files
├── train/               # PRM training, PRM server, reward, and veRL training scripts
├── eval/                # Direct inference and PRM-guided beam search
├── verl/                # veRL backend for GRPO training
├── OpenRLHF/            # OpenRLHF backend for PRM training
└── README.md
```

## Environment Setup

The project uses two training backends: **veRL** for test generator RL training and **OpenRLHF** for PRM training. 
### 1. Create the Conda Environment

```bash
conda create -n steer python=3.10 -y
conda activate steer
```

### 2. Install veRL Environment

Use this environment for generator training and evaluation.

```bash
cd verl

python -m pip install -U pip setuptools wheel

pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip install -e .[vllm] "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"

pip install --force-reinstall "ray==2.49.2"
pip install --force-reinstall "numpy<2.0.0"
pip install "transformers==4.57.6"

cd ..
```

### 3. Install OpenRLHF Environment

Use this environment for PRM training. If `OpenRLHF/` is not already included, place a PRM-compatible OpenRLHF release under this directory first.

```bash
cd OpenRLHF

python -m pip install -U pip setuptools wheel

pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

pip install -e .

cd ..
```

## Dataset Construction

The dataset construction pipeline builds step-level PRM supervision data and generator training data. It starts from programming tasks with correct and incorrect solutions, filters tasks, generates balanced positive/negative trajectories, computes MC-based step labels, applies LLM-based verification, and converts the retained consensus data into the PRM training format.

Before running the script, set the required environment variables with your own model paths and API-compatible endpoints:

```bash
export TOKENIZER_MODEL_PATH="/path/to/tokenizer_or_model"

export GENERATION_API_BASE_URL="..."
export GENERATION_API_KEY="your-generation-api-key"
export GENERATION_MODEL_NAME="your-generation-model-name"

export JUDGE_API_BASE_URL="..."
export JUDGE_API_KEY="your-judge-api-key"
export JUDGE_MODEL_NAME="your-judge-model-name"
```

Then run:

```bash
bash data_construction/run_data_construction.sh
```

By default, outputs are written to:

```text
outputs/data_construction/
```

## PRM Training

The PRM is trained on the consensus step-level labels produced by the data construction pipeline. It learns to score whether an intermediate reasoning step supports successful bug-exposing test generation.

```bash
bash train/train_prm.sh
```

Before running, update the script with the model path, processed PRM data path, and output directory used in your environment.

## Test Generator Training

The test generator is trained with GRPO using execution-based outcome rewards and PRM-based process rewards. The PRM is served separately and queried during reward computation.

A typical workflow is:

```bash
python train/prepare_verl_data.py --help
bash train/prm_server.sh
bash train/train_test_generator.sh
```

`prepare_verl_data.py` prepares data for veRL, `prm_server.sh` launches the PRM scoring service, and `run.sh` starts test generator training.



## Evaluation

The `eval/` directory provides direct inference and PRM-guided beam search.

```bash
bash eval/infer.sh
bash eval/infer_beam_search.sh
```



