#!/bin/bash
# ============================================================================
# NovaMind-3B Pre-training — SLURM batch job
#
# Submit:  sbatch scripts/pretrain_slurm.sh
# Resume:  sbatch scripts/pretrain_slurm.sh --resume /path/to/ckpt.pt
# Monitor: tail -f /mnt/zone/A/logs/novamind-pretrain-<JOBID>.out
# ============================================================================

#SBATCH --job-name=novamind-3b-pretrain
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --open-mode=append
#SBATCH --output=/mnt/zone/A/logs/novamind-pretrain-%j.out
#SBATCH --error=/mnt/zone/A/logs/novamind-pretrain-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
PYTHON=/home/shashikant/anaconda3/envs/deepfill/bin/python
PROJECT=/mnt/zone/B/GPT/deepseek-1b
CKPT_DIR=/mnt/zone/A/checkpoints/novamind-3b/pretrain
LOG_DIR=/mnt/zone/A/logs
WANDB_PROJECT=novamind-3b-pretrain
RUN_NAME="pretrain-slurm-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M)"

# Parse optional --resume flag passed to sbatch via --export or positional arg:
# e.g.  sbatch scripts/pretrain_slurm.sh --resume /mnt/zone/A/.../step_10000.pt
RESUME_CKPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_CKPT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "======================================================================"
echo " NovaMind-3B Pre-training"
echo " Job ID  : ${SLURM_JOB_ID:-local}"
echo " Run     : $RUN_NAME"
echo " Node    : $(hostname)"
echo " GPUs    : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' '|')"
echo " Started : $(date)"
echo " Resume  : ${RESUME_CKPT:-none}"
echo "======================================================================"

# ── Torchrun ─────────────────────────────────────────────────────────────────
TRAIN_CMD=(
    torchrun
        --nproc_per_node=2
        --master_port=29500
    "$PROJECT/train.py"
        --wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "$RUN_NAME"
        --output-dir "$CKPT_DIR"
)

if [[ -n "$RESUME_CKPT" ]]; then
    TRAIN_CMD+=(--resume "$RESUME_CKPT")
fi

"${TRAIN_CMD[@]}"
EXIT_CODE=$?

echo "======================================================================"
echo " Finished : $(date)  |  exit code: $EXIT_CODE"
echo "======================================================================"
exit $EXIT_CODE
