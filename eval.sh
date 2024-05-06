#!/bin/zsh
output_dir="results"
context_window=4096
batch_size=1

export TOKENIZERS_PARALLELISM="false"

for checkpoint_suffix in 6000 12000 18000 24000; do
  checkpoint_path="checkpoints/checkpoints/final-run-3/hf-ep0-ba${checkpoint_suffix}-rank0/checkpoints/final-run-3/ep0-ba${checkpoint_suffix}-rank0-hf"
  for file in data-final/eval-icl/*.jsonl; do
    echo "Evaluating $file with $checkpoint_path:"
    python eval.py --checkpoint_path "$checkpoint_path" --jsonl_file "$file" --output_dir "$output_dir" --context_window "$context_window" --batch_size "$batch_size"
  done
do