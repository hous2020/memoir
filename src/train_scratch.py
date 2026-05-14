import os
import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import evaluate

from data_loader import load_huggingface_dataset
from transformer_model import TransformerSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Hyperparameters
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 8
LR = 0.0001
MAX_SEQ_LEN = 128
GRADIENT_CLIP = 1.0
USE_LR_SCHEDULER = True

SOS_TOKEN = "<s>"
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"


def get_special_token_ids(tokenizer):
    token_ids = {
        "sos": tokenizer.token_to_id(SOS_TOKEN),
        "pad": tokenizer.token_to_id(PAD_TOKEN),
        "eos": tokenizer.token_to_id(EOS_TOKEN),
    }
    missing = [name for name, token_id in token_ids.items() if token_id is None]
    if missing:
        raise ValueError(f"Missing special token ids in tokenizer: {', '.join(missing)}")
    return token_ids


class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_ids = get_special_token_ids(tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item.get("text") or ""
        tgt_text = item.get("summary") or ""

        src_enc = self.tokenizer.encode(src_text).ids[: self.max_len]
        tgt_enc = self.tokenizer.encode(tgt_text).ids[: self.max_len - 2]
        tgt_enc = [self.special_ids["sos"], *tgt_enc, self.special_ids["eos"]]

        return {
            "src": torch.tensor(src_enc, dtype=torch.long),
            "tgt": torch.tensor(tgt_enc, dtype=torch.long),
        }


def build_collate_fn(pad_id):
    def collate_fn(batch):
        src_list = [item["src"] for item in batch]
        tgt_list = [item["tgt"] for item in batch]

        src_pad = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=pad_id)
        tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_list, padding_value=pad_id)

        return src_pad, tgt_pad

    return collate_fn


def train(
    dataset_name="xlsum",
    max_samples=None,
    tokenizer_path="data/custom_tokenizer.json",
    model_path="models/transformer_scratch.pth",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    max_seq_len=MAX_SEQ_LEN,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    gradient_clip=GRADIENT_CLIP,
    use_lr_scheduler=USE_LR_SCHEDULER,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        print(f"Tokenizer not found at {tokenizer_path}. Run src/train_tokenizer.py first.")
        return

    special_ids = get_special_token_ids(tokenizer)

    # Load training dataset
    print(f"Loading training dataset: {dataset_name}...")
    raw_train_dataset = load_huggingface_dataset(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_samples,
    )

    # Load validation dataset
    print(f"Loading validation dataset: {dataset_name}...")
    raw_val_dataset = load_huggingface_dataset(
        dataset_name=dataset_name,
        split="validation",
        max_samples=min(2000, max_samples if max_samples else 2000),
    )

    train_dataset = SummarizationDataset(raw_train_dataset, tokenizer, max_len=max_seq_len)
    val_dataset = SummarizationDataset(raw_val_dataset, tokenizer, max_len=max_seq_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(special_ids["pad"]),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(special_ids["pad"]),
    )

    model = TransformerSummarizer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=special_ids["pad"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.01
        )
        print(f"Using CosineAnnealingLR scheduler (T_max={epochs}, eta_min={learning_rate * 0.01})")

    # Load ROUGE metric for validation
    rouge_metric = evaluate.load("rouge")

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size} | Learning rate: {learning_rate}\n")

    model.train()
    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_padding_mask = (src == special_ids["pad"]).transpose(0, 1)
            tgt_padding_mask = (tgt_input == special_ids["pad"]).transpose(0, 1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)

            optimizer.zero_grad()
            output = model(
                src,
                tgt_input,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask,
            )

            loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / max(len(train_dataloader), 1)
        
        # Validation phase
        print(f"\nRunning validation...")
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_references = []

        with torch.no_grad():
            for val_src, val_tgt in tqdm(val_dataloader, desc="Validation"):
                val_src, val_tgt = val_src.to(device), val_tgt.to(device)
                
                val_tgt_input = val_tgt[:-1, :]
                val_tgt_output = val_tgt[1:, :]

                val_src_padding_mask = (val_src == special_ids["pad"]).transpose(0, 1)
                val_tgt_padding_mask = (val_tgt_input == special_ids["pad"]).transpose(0, 1)
                val_tgt_mask = model.generate_square_subsequent_mask(val_tgt_input.size(0)).to(device)

                val_output = model(
                    val_src,
                    val_tgt_input,
                    src_padding_mask=val_src_padding_mask,
                    tgt_padding_mask=val_tgt_padding_mask,
                    tgt_mask=val_tgt_mask,
                )

                v_loss = criterion(val_output.view(-1, val_output.size(-1)), val_tgt_output.reshape(-1))
                val_loss += v_loss.item()

                # Generate predictions for ROUGE (limit to avoid slow validation)
                max_rouge_samples = 100
                for i in range(min(val_src.size(1), max(0, max_rouge_samples - len(val_predictions)))):
                    try:
                        generated = generate_summary(
                            model, val_src[:, i:i+1], special_ids, max_len=max_seq_len
                        )
                        reference = tokenizer.decode(
                            [t.item() for t in val_tgt[:, i] if t.item() != special_ids["pad"]],
                            skip_special_tokens=True
                        )
                        if generated.strip() and reference.strip():
                            val_predictions.append(generated)
                            val_references.append(reference)
                    except Exception as gen_err:
                        logger.warning(f"Generation error for sample {i}: {gen_err}")

        avg_val_loss = val_loss / max(len(val_dataloader), 1)
        
        # Compute ROUGE scores
        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if val_predictions:
            rouge_scores = rouge_metric.compute(
                predictions=val_predictions, 
                references=val_references
            )

        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = learning_rate

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs} Complete")
        print(f"{'='*60}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f} | ROUGE-2: {rouge_scores['rouge2']:.4f} | ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            config = {
                "vocab_size": tokenizer.get_vocab_size(),
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "max_seq_len": max_seq_len,
            }
            config_path = os.path.splitext(model_path)[0] + "_config.json"
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2)
            
            print(f"✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        model.train()
        print()

    config_path = os.path.splitext(model_path)[0] + "_config.json"
    print(f"\nTraining complete. Best model saved to {model_path}.")
    if os.path.exists(config_path):
        print(f"Model config saved to {config_path}.")


def generate_summary(model, src_tensor, special_ids, max_len=128):
    """
    Generate a summary using greedy decoding.
    
    Args:
        model: TransformerSummarizer model
        src_tensor: Source tensor [seq_len, 1]
        special_ids: Dictionary with 'sos', 'eos', 'pad' token IDs
        max_len: Maximum generation length
    
    Returns:
        Generated token IDs list
    """
    device = src_tensor.device
    tgt_indices = [special_ids["sos"]]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(0)).to(device)
            
            output = model(src_tensor, tgt, tgt_mask=tgt_mask)
            
            # Get last token prediction
            next_token = output[-1, 0, :].argmax().item()
            
            if next_token == special_ids["eos"]:
                break
            
            tgt_indices.append(next_token)
    
    return tgt_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Train the scratch Transformer summarizer.")
    parser.add_argument(
        "--dataset-name",
        choices=["xlsum", "mlsum", "orangesum_abstract", "orangesum_title", "orangesum_wikilead", "all"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokenizer-path", default="data/custom_tokenizer.json")
    parser.add_argument("--model-path", default="models/transformer_scratch.pth")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--gradient-clip", type=float, default=GRADIENT_CLIP)
    parser.add_argument("--no-lr-scheduler", action="store_true", help="Disable LR scheduler")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        gradient_clip=args.gradient_clip,
        use_lr_scheduler=not args.no_lr_scheduler,
    )
