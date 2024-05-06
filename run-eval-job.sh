#!/bin/bash
#SBATCH --job-name=eval_task
#SBATCH --output=eval_task_%A_%a.out
#SBATCH --error=eval_task_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4

checkpoint_suffixes=(6000 12000 18000 24000 30000)
checkpoint_suffix=${checkpoint_suffixes[$SLURM_ARRAY_TASK_ID]}
output_dir="results" context_window=4096 batch_size=1
export TOKENIZERS_PARALLELISM="false"

checkpoint_path="checkpoints/final-run-3/hf-ep0-ba${checkpoint_suffix}-rank0/checkpoints/final-run-3/ep0-ba${checkpoint_suffix}-rank0-hf"
for file in data-final/eval-icl/*.jsonl;
do
  echo "Evaluating $file with $checkpoint_path:"
  srun bash -c "source ~/.bashrc && conda init && conda activate pytorch && python eval.py --checkpoint_path '$checkpoint_path' --jsonl_file '$file' --output_dir '$output_dir' --context_window '$context_window' --batch_size '$batch_size'"
done
