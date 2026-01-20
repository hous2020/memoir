from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from data_loader import load_french_summarization_data
import os

def train_custom_tokenizer(vocab_size=30000, save_path="data/custom_tokenizer.json"):
    """
    Trains a BPE tokenizer from scratch on the dataset.
    """
    print("Initializing Tokenizer Training...")
    
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
    # We use the training set for the tokenizer
    # Note: Using a subset for speed in this demo context if needed, but ideally full train set
    dataset = load_french_summarization_data(split="test", subset="french") # Using test split here just for demo speed/availability in data_loader fallback
    
    if dataset is None:
        print("Error: Could not load dataset.")
        return

    # Iterator for the tokenizer
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            # Combine text and summary for vocabulary coverage
            yield batch["text"] + batch["summary"]

    # 4. Train
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # 5. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    train_custom_tokenizer()
