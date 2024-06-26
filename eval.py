import os
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from collections import defaultdict
from model import load_pretrained_model
import matplotlib.pyplot as plt


def icl_tokenize(tokenizer, context_window=4096):
    def icl_tokenize_inner(examples):
        context_indices = tokenizer.encode(examples["context"])
        continuation_indices = tokenizer.encode(examples["continuation"])

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_tokenize_inner


def icl_collate_fn(tokenizer):
    def icl_collate_fn_inner(examples):
        context_indices = [example["context_indices"] for example in examples]
        continuation_indices = [example["continuation_indices"] for example in examples]

        context_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(context_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        continuation_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(continuation_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_collate_fn_inner


def evaluate(
    checkpoint_path, jsonl_file, output_dir, context_window=4096, batch_size=16
):
    device = "cpu"

    tokenizer_file = checkpoint_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_file, max_len=context_window, padding_side="right", truncation=True
    )

    dataset = load_dataset(
        "json",
        data_files={"eval": jsonl_file},
        cache_dir=os.path.join(os.path.dirname(jsonl_file), ".cache"),
    ).map(icl_tokenize(tokenizer, context_window=context_window))["eval"]

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=icl_collate_fn(tokenizer),
        pin_memory=True,
        num_workers=cpu_count(),
    )

    model = load_pretrained_model(checkpoint_path, device)


    space_token_id = tokenizer.convert_tokens_to_ids(" ")
    star_token_id = tokenizer.convert_tokens_to_ids("*")

    stop_token_counts = defaultdict(int)
    generated_star_counts = defaultdict(int)
    correct_predictions = 0
    total_predictions = 0
    star_differences = []
    stats = {}

    for data in tqdm(dataloader):
        generated_token_ids = []
        input_ids = data["context_indices"].to(device)
        stopped_at = input_ids.size(1)

        while True:
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                generated_token_id = torch.argmax(outputs.logits[0, -1]).item()
                generated_token_ids.append(generated_token_id)

                if (
                    generated_token_id != space_token_id
                    and generated_token_id != star_token_id
                ):
                    stopped_at = generated_token_id
                    break

                input_ids = torch.cat(
                    (input_ids, torch.tensor([[generated_token_id]]).to(device)),
                    dim=1,
                )

        # Update stop token counts
        stop_token_counts[stopped_at] += 1

        # Count the number of stars in the generated token IDs
        generated_stars = generated_token_ids.count(star_token_id)
        generated_star_counts[generated_stars] += 1

        # Count the number of stars in the continuation token IDs
        actual_stars = data["continuation_indices"][0].tolist().count(star_token_id)

        if generated_stars == actual_stars:
            correct_predictions += 1
        total_predictions += 1

        star_difference = abs(generated_stars - actual_stars)
        star_differences.append(star_difference)

    # Compute stop token statistics
    for token_id, count in stop_token_counts.items():
        percentage = count / total_predictions * 100
        print(f"Stopped at token ID {token_id}: {percentage:.2f}%")
        stats[f"stop_token_counts/{token_id}"] = percentage

    # Compute generated star length statistics
    for star_length, count in generated_star_counts.items():
        percentage = count / total_predictions * 100
        print(f"Generated {star_length} stars: {percentage:.2f}% of the time")
        stats[f"generated_star_counts/{star_length}"] = percentage

    # Compute accuracy and mean star difference
    accuracy = correct_predictions / total_predictions
    mean_star_difference = sum(star_differences) / len(star_differences)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Star Difference: {mean_star_difference:.2f}")

    stats["accuracy"] = accuracy
    stats["mean_star_difference"] = mean_star_difference


    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint_folder = os.path.basename(checkpoint_path)
    os.makedirs(f"{output_dir}/{last_checkpoint_folder}", exist_ok=True)
    output_dir = os.path.join(output_dir, last_checkpoint_folder)

    # Save evaluation results to a CSV file
    csv_file = os.path.join(
        output_dir,
        f"eval_results_{os.path.basename(jsonl_file.replace('.jsonl', ''))}.csv",
    )

    with open(csv_file, "w") as f:
        f.write("metric,value\n")
        for metric, value in stats.items():
            f.write(f"{metric},{value}\n")


    # Create matplotlib figures for stop token counts and generated star counts
    fig_stop_tokens = plt.figure(figsize=(7, 5))
    plt.bar(stop_token_counts.keys(), stop_token_counts.values())
    plt.xlabel("Token ID")
    plt.ylabel("Count")
    plt.title("Stop Token Counts")

    fig_generated_stars = plt.figure(figsize=(7, 5))
    plt.bar(generated_star_counts.keys(), generated_star_counts.values())
    plt.xlabel("Number of Stars")
    plt.ylabel("Count")
    plt.title("Generated Star Counts")

    # save figures to output directory
    fig_stop_tokens_file = os.path.join(
        output_dir,
        f"stop_token_counts_{os.path.basename(jsonl_file.replace('.jsonl', ''))}.png",
    )
    fig_generated_stars_file = os.path.join(
        output_dir,
        f"generated_star_counts_{os.path.basename(jsonl_file.replace('.jsonl', ''))}.png",
    )

    fig_stop_tokens.savefig(fig_stop_tokens_file)
    fig_generated_stars.savefig(fig_generated_stars_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path to the checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--jsonl_file",
        type=str,
        required=True,
        help="The JSONL file to use for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the evaluation results.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=4096,
        help="The context window size for tokenization.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The batch size for evaluation."
    )

    args = parser.parse_args()

    evaluate(
        args.checkpoint_path,
        args.jsonl_file,
        args.output_dir,
        args.context_window,
        args.batch_size,
    )
