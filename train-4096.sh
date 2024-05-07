. ~/.bashrc && conda activate pytorch && composer -n 1 train.py --run_name FINAL-4096 --eval_interval 1500ba --learning_rate 1e-4 --batch_size 8 --context_window 4096 --datafolder data-final
