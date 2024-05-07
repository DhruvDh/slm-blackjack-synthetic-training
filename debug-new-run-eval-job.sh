#!/bin/bash

checkpoint_dirs=(
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-2048"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-4096"
  "/users/ddhamani/8156/slm-blackjack-synthetic-training/checkpoints/FINAL-8192"
)
output_dir="results" batch_size=1
export TOKENIZERS_PARALLELISM="false"

processed_file="processed_checkpoints.txt"
touch "$processed_file" # Create the processed_file if it doesn't exist

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
  context_window="${checkpoint_dir##*-}"
  for checkpoint_file in "$checkpoint_dir"/*.pt; do
    if [[ $checkpoint_file != *"latest-rank0.pt" ]]; then
      for jsonl_file in data-final/eval-icl/*.jsonl; do
        if ! grep -q "$checkpoint_file,$jsonl_file" "$processed_file"; then
          echo "Evaluating $jsonl_file with $checkpoint_file:"
          eval_command="python new-eval.py --checkpoint_path '$checkpoint_file' --jsonl_file '$jsonl_file' --output_dir '$output_dir' --context_window '$context_window' --batch_size '$batch_size' --use_gpu"
          echo "Command: $eval_command"

          # Write the processed checkpoint and jsonl file to the processed_file
          echo "$checkpoint_file,$jsonl_file" >>"$processed_file"
        fi
      done
    fi
  done
done
