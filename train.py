import os
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
from composer.utils import reproducibility
from composer import Trainer
from composer.core import Evaluator
from composer.loggers import FileLogger, TensorboardLogger
from composer import Callback, Event, Logger, State
import schedulefree
from model import create_model, save_checkpoint_as_hf_model

reproducibility.configure_deterministic_mode()
reproducibility.seed_all(42)


def create_sliding_windows(tokenizer, context_window=8192):
    def create_sliding_windows_inner(examples):
        input_ids = []
        labels = []

        for text in examples["text"]:
            encoded_text = tokenizer.encode(text)
            window_size = min(context_window, len(encoded_text))

            for i in range(1, window_size - 1):
                input_ids.append(encoded_text[0:i])
                labels.append(encoded_text[i + 1])

        return {"input_ids": input_ids, "labels": labels}

    return create_sliding_windows_inner


class CheckpointConverter(Callback):
    def __init__(self, run_name):
        self.run_name = run_name
        self.converted_checkpoints = {}

    def run_event(self, event: Event, state: State, logger: Logger):
        if (
            event == Event.BATCH_CHECKPOINT
            or event == Event.EPOCH_CHECKPOINT
            or event == Event.ITERATION_CHECKPOINT
        ):
            for checkpoint_path in os.listdir(f"checkpoints/{self.run_name}"):
                if checkpoint_path.endswith(".pt"):
                    hf_model_path = f"checkpoints/{self.run_name}/hf-{checkpoint_path.replace('.pt', '')}"
                    if (
                        checkpoint_path not in self.converted_checkpoints
                        or not os.path.exists(hf_model_path)
                    ):
                        save_checkpoint_as_hf_model(
                            f"checkpoints/{self.run_name}/{checkpoint_path}",
                            hf_model_path,
                        )
                        self.converted_checkpoints[checkpoint_path] = True


def train(
    run_name,
    eval_interval,
    learning_rate,
    batch_size,
    context_window,
    datafolder,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_file = f"{datafolder}/overall-tokenizer.json"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_file, padding_side="left"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=context_window
    )

    dataset = load_dataset(
        "text",
        data_files={
            "train": [f"{datafolder}/train/*.txt"],
            "eval": [f"{datafolder}/eval/*.txt"],
        },
        sample_by="document",
        keep_linebreaks=True,
        cache_dir=f"{datafolder}/.cache",
    )

    eval_dataset = dataset["eval"].map(
        create_sliding_windows(
            tokenizer=tokenizer,
            context_window=context_window,
        ),
        batched=True,
        batch_size=2,
        num_proc=cpu_count(),
        remove_columns=["text"],
    )
    print(eval_dataset)

    train_dataset = dataset["train"].map(
        create_sliding_windows(
            tokenizer=tokenizer,
            context_window=context_window,
        ),
        batched=True,
        batch_size=2,
        num_proc=cpu_count(),
        remove_columns=["text"],
    )
    print(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=data_collator,
        prefetch_factor=int(batch_size / 8),
        num_workers=cpu_count(),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=data_collator,
        prefetch_factor=int(batch_size / 8),
        num_workers=cpu_count(),
    )

    ppl_eval = Evaluator(
        label="ppl_eval",
        dataloader=eval_dataloader,
        metric_names=["LanguagePerplexity"],
    )

    model = create_model(tokenizer, context_window, device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=[ppl_eval],
        eval_interval=eval_interval,
        max_duration="42000ba",
        save_folder=f"checkpoints/{run_name}",
        save_interval=eval_interval,
        save_overwrite=False,
        device_train_microbatch_size="auto",
        device="gpu",
        run_name=run_name,
        autoresume=True,
        precision="amp_bf16",
        console_log_interval=eval_interval,
        callbacks=[CheckpointConverter(run_name)],
        loggers=[
            FileLogger(f"checkpoints/{run_name}_logs.txt"),
            TensorboardLogger(),
        ],
    )

    trainer.fit()
    trainer.close()


if __name__ == "__main__":
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["NODE_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["LOCAL_WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "29500"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # torch.distributed.init_process_group()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import argparse

    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument(
        "--run_name", type=str, required=True, help="The name of the training run."
    )
    parser.add_argument(
        "--eval_interval",
        type=str,
        required=True,
        help="The evaluation interval, e.g., '6000ba' for every 6000 batches.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="The learning rate for the optimizer.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="The batch size for training."
    )
    parser.add_argument(
        "--context_window",
        type=int,
        required=True,
        help="The context window size for the sliding window approach.",
    )
    parser.add_argument(
        "--datafolder",
        type=str,
        required=True,
        help="The path to the data folder containing the training and evaluation data.",
    )

    args = parser.parse_args()

    train(
        args.run_name,
        args.eval_interval,
        args.learning_rate,
        args.batch_size,
        args.context_window,
        args.datafolder,
    )
