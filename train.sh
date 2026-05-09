#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29502}
DATA_ROOT=${DATA_ROOT:-/path/to/imagenet}

model="vic_ti_224_gelu"
batch_size=128
scheduler="cosine"
epochs=400
opt="adamw"
opt_eps=1e-8
workers=4
warmup_lr=0.000001
warmup_epochs=20
remode="pixel"
reprob=0.25
weight_decay=5e-2
lr=0.002
drop=0
drop_path=0.1
mixup=0.8
cutmix=1.0
model_ema_decay=0.99996
aa="rand-m9-mstd0.5-inc1"
color_jitter=0.4
output_path="outputs_imagenet1000_v4_${epochs}_${lr}"
num_classes=1000

k=9
num_hyperedges=16
laplacian_alpha=0.5
use_structure_induction=true
e2v_ratio=2.0
epsilon=0.1
enable_hyperedge_dropout=false
e_drop_rate=0.2
clip_grad=0.1
seed=42

TRAIN_ARGS=(
    "${DATA_ROOT}"
    --model "${model}"
    --num-classes "${num_classes}"
    --sched "${scheduler}"
    --epochs "${epochs}"
    --opt "${opt}"
    -j "${workers}"
    --warmup-lr "${warmup_lr}"
    --mixup "${mixup}"
    --cutmix "${cutmix}"
    --model-ema
    --model-ema-decay "${model_ema_decay}"
    --aa "${aa}"
    --color-jitter "${color_jitter}"
    --warmup-epochs "${warmup_epochs}"
    --opt-eps "${opt_eps}"
    --repeated-aug
    --remode "${remode}"
    --reprob "${reprob}"
    --amp
    --lr "${lr}"
    --weight-decay "${weight_decay}"
    --drop "${drop}"
    --drop-path "${drop_path}"
    --clip-grad "${clip_grad}"
    -b "${batch_size}"
    --output "${output_path}"
    --num-gpu "${NPROC_PER_NODE}"
    --k "${k}"
    --relative-pos
    --num-hyperedges "${num_hyperedges}"
    --laplacian-alpha "${laplacian_alpha}"
    --use-structure-induction "${use_structure_induction}"
    --e2v-ratio "${e2v_ratio}"
    --use-stochastic
    --epsilon "${epsilon}"
    --e-drop-rate "${e_drop_rate}"
    --seed "${seed}"
)

if [[ "${enable_hyperedge_dropout}" == "true" ]]; then
    TRAIN_ARGS+=(--hyperedge-dropout)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py "${TRAIN_ARGS[@]}"
