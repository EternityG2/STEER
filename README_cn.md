# STEER：基于过程监督训练与奖励引导搜索的 Bug-Exposing Test Generation

[English README](./README.md)

STEER 是一个用于生成 bug-exposing test case 的过程监督框架。给定编程题目描述和错误代码，模型需要生成一个测试用例，使其既能触发错误代码的 bug，又能给出正确的期望输出。STEER 使用任务相关的 Process Reward Model（PRM）在两个阶段发挥作用：训练阶段提供过程奖励，推理阶段引导候选推理路径搜索。

整个流程包括四部分：数据集构建、PRM 训练、Test Generator 训练和评估。

## 项目结构

```text
STEER/
├── data_construction/   # 构建 PRM 数据和生成器训练数据
├── datasets/            # 本地数据目录
├── train/               # PRM 训练、PRM 服务、奖励函数和 veRL 训练脚本
├── eval/                # 普通推理和 PRM-guided beam search
├── verl/                # 用于 GRPO 训练的 veRL 后端
├── OpenRLHF/            # 用于 PRM 训练的 OpenRLHF 后端
└── README.md
```

## 环境配置

项目主要依赖两个训练后端：**veRL** 用于 Test Generator 的强化学习训练，**OpenRLHF** 用于 PRM 训练。
### 1. 创建 Conda 环境

```bash
conda create -n steer python=3.10 -y
conda activate steer
```

### 2. 安装 veRL 环境

该环境用于生成器训练和评估。

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

### 3. 安装 OpenRLHF 环境

该环境用于 PRM 训练。如果当前仓库下没有 `OpenRLHF/`，请先将支持 PRM 训练的 OpenRLHF 版本放到该目录下。

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

## 数据集构建

数据构建流程用于生成 PRM 的 step-level 监督数据以及 Test Generator 的训练数据。整体流程包括任务过滤、正负样本构建、MC-based step labeling、LLM-based verification，以及将共识过滤后的数据转换为 OpenRLHF PRM 训练格式。

运行前需要先设置模型路径和 API-compatible endpoint。下面只保留占位符，请替换为自己的本地配置：

```bash
export TOKENIZER_MODEL_PATH="/path/to/tokenizer_or_model"

export GENERATION_API_BASE_URL="..."
export GENERATION_API_KEY="your-generation-api-key"
export GENERATION_MODEL_NAME="your-generation-model-name"

export JUDGE_API_BASE_URL="..."
export JUDGE_API_KEY="your-judge-api-key"
export JUDGE_MODEL_NAME="your-judge-model-name"
```

然后运行：

```bash
bash data_construction/run_data_construction.sh
```

默认输出目录为：

```text
outputs/data_construction/
```

## PRM 训练

PRM 使用数据构建阶段得到的共识 step-level 标签进行训练，用于判断中间推理步骤是否有助于生成成功的 bug-exposing test case。

```bash
bash train/train_prm.sh
```

运行前需要在脚本中修改模型路径、PRM 数据路径和输出目录。

## Test Generator 训练

Test Generator 使用 GRPO 训练，奖励由执行结果奖励和 PRM 过程奖励组成。训练时 PRM 会作为独立服务启动，并在 reward 计算时被调用。

典型流程如下：

```bash
python train/prepare_verl_data.py --help
bash train/prm_server.sh
bash train/train_test_generator.sh
```

其中，`prepare_verl_data.py` 用于准备 veRL 格式数据，`prm_server.sh` 用于启动 PRM 打分服务，`run.sh` 用于启动 Test Generator 训练。

## 评估

`eval/` 目录提供普通推理和 PRM-guided beam search 两种评估方式。

```bash
bash eval/infer.sh
bash eval/infer_beam_search.sh
```


