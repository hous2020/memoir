import argparse
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from data_loader import load_huggingface_dataset


def train_custom_tokenizer(
    vocab_size=30000,
    save_path="data/custom_tokenizer.json",
    dataset_name="xlsum",
    max_samples=None,
):
    """
    Trains a BPE tokenizer from scratch on all combined datasets.
    """
    print("Initializing tokenizer training from Hugging Face datasets...")
    
    # 1. Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # 2. Prepare trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 3. Load data
    dataset = load_huggingface_dataset(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_samples,
    )

    # Iterator for the tokenizer
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            # Combine text and summary for vocabulary coverage
            texts = [text for text in batch["text"] if text]
            summaries = [summary for summary in batch["summary"] if summary]
            yield texts + summaries

    # 4. Train
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # 5. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for summarization.")
    parser.add_argument(
        "--dataset-name",
        choices=["xlsum", "mlsum", "wiki_lingua_fr", "all"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--save-path", default="data/custom_tokenizer.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_tokenizer(
        vocab_size=args.vocab_size,
        save_path=args.save_path,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
    )
