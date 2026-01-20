import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from transformer_model import TransformerSummarizer
from data_loader import load_french_summarization_data
import os

# Hyperparameters
VOCAB_SIZE = 30000
D_MODEL = 256  # Reduced for demo/speed (standard is 512)
NHEAD = 4      # Reduced (standard is 8)
NUM_LAYERS = 2 # Reduced (standard is 6)
BATCH_SIZE = 4 # Small batch size for CPU/Demo
EPOCHS = 1     # Just 1 epoch for testing pipeline
LR = 0.0001
MAX_SEQ_LEN = 128

class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['text']
        tgt_text = item['summary']

        # Tokenize
        src_enc = self.tokenizer.encode(src_text).ids
        tgt_enc = self.tokenizer.encode(tgt_text).ids

        # Truncate
        src_enc = src_enc[:self.max_len]
        tgt_enc = tgt_enc[:self.max_len]

        # Add SOS/EOS tokens (assuming tokenizer adds them or handled here)
        # BPE usually doesn't add special tokens by default unless configured
        # Let's verify special token IDs in a real scenario. 
        # For this demo, we assume simple IDs or raw.
        # Ideally: [SOS] + seq + [EOS]
        
        return {
            "src": torch.tensor(src_enc, dtype=torch.long),
            "tgt": torch.tensor(tgt_enc, dtype=torch.long)
        }

def collate_fn(batch):
    src_list = [item['src'] for item in batch]
    tgt_list = [item['tgt'] for item in batch]
    
    # Pad sequences
    src_pad = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=1) # Assuming 1 is pad
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_list, padding_value=1)
    
    return src_pad, tgt_pad

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("data/custom_tokenizer.json")
    except:
        print("Tokenizer not found. Run src/train_tokenizer.py first.")
        return

    # 2. Load Data
    raw_dataset = load_french_summarization_data(split="test", subset="french")
    if raw_dataset is None: return
    
    dataset = SummarizationDataset(raw_dataset, tokenizer, max_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. Initialize Model
    model = TransformerSummarizer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=1) # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Transformer expects tgt input to be shifted
            tgt_input = tgt[:-1, :] # Input to decoder
            tgt_output = tgt[1:, :] # Expected output
            
            # Create masks
            src_padding_mask = (src == 1).transpose(0, 1) # [batch, seq_len]
            tgt_padding_mask = (tgt_input == 1).transpose(0, 1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(src, tgt_input, 
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_padding_mask,
                           tgt_mask=tgt_mask)
            
            # Reshape for loss
            # Output: [seq_len, batch, vocab_size] -> [seq_len*batch, vocab_size]
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # 5. Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_scratch.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
