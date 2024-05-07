#!/bin/bash
#SBATCH --job-name=eval_task
#SBATCH --output=eval_task_%A_%a.out
#SBATCH --error=eval_task_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

checkpoint_dirs=(
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-2048"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-4096"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-8192"
)
output_dir="results" batch_size=1
export TOKENIZERS_PARALLELISM="false"

processed_file="processed_checkpoints.txt"

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
  context_window="${checkpoint_dir##*-}"
  for checkpoint_file in "$checkpoint_dir"/*.pt; do
    if [[ $checkpoint_file != *"latest-rank0.pt" ]]; then
      for jsonl_file in data-final/eval-icl/*.jsonl; do
        if ! grep -q "$checkpoint_file,$jsonl_file" "$processed_file"; then
          echo "Evaluating $jsonl_file with $checkpoint_file:"
          srun bash -c "source ~/.bashrc && conda init && conda activate pytorch && python new-eval.py --checkpoint_path '$checkpoint_file' --jsonl_file '$jsonl_file' --output_dir '$output_dir' --context_window '$context_window' --batch_size '$batch_size' --use_gpu"

          # Write the processed checkpoint and jsonl file to the processed_file
          echo "$checkpoint_file,$jsonl_file" >>"$processed_file"
        fi
      done
    fi
  done
done
