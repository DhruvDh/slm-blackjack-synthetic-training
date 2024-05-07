#!/bin/bash
#SBATCH --job-name=eval_task
#SBATCH --output=eval_task_%A_%a.out
#SBATCH --error=eval_task_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=GPU

checkpoint_dirs=(
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-2048"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-4096"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-8192"
)
output_dir="results" batch_size=1
export TOKENIZERS_PARALLELISM="false"

processed_file="processed_checkpoints.txt"
touch "$processed_file" # Create the processed_file if it doesn't exist

eval_batches=(3000 6000 9000)

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
  context_window="${checkpoint_dir##*-}"
  for batch in "${eval_batches[@]}"; do
    checkpoint_file="$checkpoint_dir/ep0-ba${batch}-rank0.pt"
    if [[ -f "$checkpoint_file" ]]; then
      for jsonl_file in data-final/eval-icl/*.jsonl; do
        if ! grep -q "$checkpoint_file,$jsonl_file" "$processed_file"; then
          echo "Evaluating $jsonl_file with $checkpoint_file:"
          srun bash -c "source ~/.bashrc && conda init && conda activate pytorch && python new-eval.py --checkpoint_path '$checkpoint_file' --jsonl_file '$jsonl_file' --output_dir '$output_dir' --context_window '$context_window' --batch_size '$batch_size' --use_gpu"

          # Write the processed checkpoint and jsonl file to the processed_file
          echo "$checkpoint_file,$jsonl_file" >>"$processed_file"

          rm -rf tokenizer-save-dir-*
        fi
      done
    fi
  done
done
