#!/usr/bin/env bash
set -x

# ===== paths =====
MODEL_PATH=""
TRAIN_FILE=""
SAVE_PATH=""

# ===== training command =====
read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path ${SAVE_PATH} \
   --save_steps 500 \
   --logging_steps 10 \
   --eval_steps 500 \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --pretrain ${MODEL_PATH} \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 1e-6 \
   --dataset ${TRAIN_FILE} \
   --input_key input \
   --label_key label \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --packing_samples \
   --load_checkpoint \
   --placeholder_token ки \
   --reward_tokens + - \
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
