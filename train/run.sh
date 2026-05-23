#!/bin/bash

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export USE_DUAL_OBJECTIVE=true

export TMPDIR=""
export RAY_TMPDIR=""

MODEL_PATH=""
DATA_PATH=""
VALIDATION_DATA_PATH=""
REWARD_MODULE="compiler_reward"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-prm_grpo}"

RAY_TMP_DIR=""
RAY_SPILL_DIR=""

mkdir -p "$RAY_TMP_DIR"
mkdir -p "$RAY_SPILL_DIR"

echo "Starting Ray"
ray stop --force
ray start --head \
    --num-gpus=4 \
    --port=6379 \
    --temp-dir="$RAY_TMP_DIR" \
    --object-spilling-directory="$RAY_SPILL_DIR"

echo "Starting GRPO training"


python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH \
    data.val_files=$VALIDATION_DATA_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.load_format="safetensors" \
    \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$(pwd)/compiler_reward.py \
    custom_reward_function.name=compute_score \
    \
    algorithm.adv_estimator=grpo_dual\
    +algorithm.lambda_process=0.5\
    algorithm.use_kl_in_reward=False \
    \
    trainer.logger='["tensorboard"]' \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=./verl_checkpoints/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \