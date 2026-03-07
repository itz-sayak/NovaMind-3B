#!/bin/bash
# ============================================================================
# NovaMind-3B Pre-training — interactive srun launcher
#
# Usage:
#   bash scripts/pretrain_srun.sh                       # fresh run
#   bash scripts/pretrain_srun.sh --resume /path/ckpt   # resume
#
# Output is tee'd to /mnt/zone/A/logs/novamind-pretrain-<timestamp>.log
# so you can safely detach (e.g. via tmux/screen) and tail the log file.
# ============================================================================

set -euo pipefail

PYTHON=/home/shashikant/anaconda3/envs/deepfill/bin/python
PROJECT=/mnt/zone/B/GPT/deepseek-1b
CKPT_DIR=/mnt/zone/A/checkpoints/novamind-3b/pretrain
LOG_DIR=/mnt/zone/A/logs
WANDB_PROJECT=novamind-3b-pretrain
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="pretrain-interactive-${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/novamind-pretrain-${TIMESTAMP}.log"

# Passthrough flags (e.g. --resume /path/to/ckpt.pt)
EXTRA_ARGS=("$@")

mkdir -p "$LOG_DIR" "$CKPT_DIR"

cat <<EOF
======================================================================
 NovaMind-3B Pre-training  (interactive srun)
 Run     : $RUN_NAME
 Log     : $LOG_FILE
 Started : $(date)
 Args    : ${EXTRA_ARGS[*]:-none}
======================================================================
EOF

# Allocate 2 GPUs on partition gpu2 and launch torchrun inside the srun job
srun \
    --partition=gpu2 \
    --gres=gpu:2 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --time=72:00:00 \
    --pty \
    bash -c "
        torchrun \
            --nproc_per_node=2 \
            --master_port=29500 \
            '$PROJECT/train.py' \
            --wandb \
            --wandb-project '$WANDB_PROJECT' \
            --wandb-run-name '$RUN_NAME' \
            --output-dir '$CKPT_DIR' \
            ${EXTRA_ARGS[*]:-}
    " 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "======================================================================"
echo " Finished : $(date)  |  exit code: $EXIT_CODE"
echo "======================================================================"
exit $EXIT_CODE
