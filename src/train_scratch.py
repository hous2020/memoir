import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from data_loader import load_huggingface_dataset
from transformer_model import TransformerSummarizer


# Hyperparameters
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 1
LR = 0.0001
MAX_SEQ_LEN = 128

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


def train(dataset_name="xlsum", max_samples=None, tokenizer_path="data/custom_tokenizer.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        print(f"Tokenizer not found at {tokenizer_path}. Run src/train_tokenizer.py first.")
        return

    special_ids = get_special_token_ids(tokenizer)

    raw_dataset = load_huggingface_dataset(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_samples,
    )

    dataset = SummarizationDataset(raw_dataset, tokenizer, max_len=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=build_collate_fn(special_ids["pad"]),
    )

    model = TransformerSummarizer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=special_ids["pad"])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_idx, (src, tgt) in enumerate(dataloader):
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
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1} complete | Average loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_scratch.pth")
    print("Training complete. Model saved.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the scratch Transformer summarizer.")
    parser.add_argument("--dataset-name", choices=["xlsum", "mlsum", "all"], default="xlsum")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tokenizer-path", default="data/custom_tokenizer.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        tokenizer_path=args.tokenizer_path,
    )
