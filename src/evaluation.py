import argparse

import evaluate
from tqdm import tqdm

from data_loader import load_huggingface_dataset, load_local_dataset
from summarizer import FrenchSummarizer


def evaluate_model(
    num_samples=20,
    model_type="scratch",
    dataset_name="xlsum",
    dataset_path=None,
):
    """
    Evaluates a summarization model with ROUGE metrics.

    Args:
        num_samples: Number of test samples to evaluate.
        model_type: "scratch".
        dataset_name: Hugging Face dataset name: "xlsum" or "mlsum".
        dataset_path: Optional local Hugging Face dataset path.
    """
    if dataset_path:
        print(f"Loading local dataset for evaluation from {dataset_path}...")
        full_dataset = load_local_dataset(dataset_path)
        if full_dataset is None:
            return None
        if "test" not in full_dataset:
            raise ValueError(f"Dataset at {dataset_path} does not contain a test split.")
        dataset = full_dataset["test"].select(range(min(num_samples, len(full_dataset["test"]))))
    else:
        dataset = load_huggingface_dataset(
            dataset_name=dataset_name,
            split="test",
            max_samples=num_samples,
        )

    if len(dataset) == 0:
        raise ValueError("The selected evaluation dataset is empty.")

    model_path = "models/transformer_scratch.pth" if model_type == "scratch" else None
    summarizer = FrenchSummarizer(model_type=model_type, model_path=model_path)
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    print(f"Generating summaries for {len(dataset)} samples...")
    for item in tqdm(dataset):
        text = item.get("text") or ""
        reference_summary = item.get("summary") or ""
        generated_summary = summarizer.summarize(text)

        predictions.append(generated_summary)
        references.append(reference_summary)

    results = rouge.compute(predictions=predictions, references=references)

    print("\nEvaluation Results (ROUGE):")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\n--- Example ---")
    print(f"Original Summary: {references[0]}")
    print(f"Generated Summary: {predictions[0]}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the scratch summarization model with ROUGE.")
    parser.add_argument("--model-type", choices=["scratch"], default="scratch")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--dataset-name", choices=["xlsum", "mlsum"], default="xlsum")
    parser.add_argument("--dataset-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n--- Evaluating scratch model ---")
    evaluate_model(
        num_samples=args.num_samples,
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )
