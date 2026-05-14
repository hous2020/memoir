import logging
import os
import json
from typing import Optional

import torch
from tokenizers import Tokenizer

from transformer_model import TransformerSummarizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FrenchSummarizer:
    def __init__(self, model_type: str = "scratch", model_path: Optional[str] = None):
        """
        Initializes the French summarizer with the custom Transformer model.

        Args:
            model_type: Only "scratch" is supported.
            model_path: Path to the scratch model weights.
        """
        if model_type != "scratch":
            raise ValueError("Only model_type='scratch' is supported in this project.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = "scratch"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading custom model from scratch: {model_path}...")

        self.custom_tokenizer = Tokenizer.from_file("data/custom_tokenizer.json")
        self.special_ids = self._load_special_ids()

        config = self._load_model_config(model_path)

        self.model = TransformerSummarizer(
            vocab_size=self.custom_tokenizer.get_vocab_size(),
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            logger.info("Custom model weights loaded successfully.")
        else:
            logger.warning(f"Model path {model_path} not found. Using untrained model.")

        self.model.eval()

    def _load_model_config(self, model_path):
        default_config = {
            "d_model": 256,
            "nhead": 4,
            "num_layers": 2,
            "max_seq_len": 128,
        }
        if not model_path:
            return default_config

        config_path = os.path.splitext(model_path)[0] + "_config.json"
        if not os.path.exists(config_path):
            logger.warning(f"Model config not found at {config_path}. Using default architecture.")
            return default_config

        with open(config_path, "r", encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)
        return {**default_config, **loaded_config}

    def _load_special_ids(self):
        special_ids = {
            "sos": self.custom_tokenizer.token_to_id("<s>"),
            "pad": self.custom_tokenizer.token_to_id("<pad>"),
            "eos": self.custom_tokenizer.token_to_id("</s>"),
            "unk": self.custom_tokenizer.token_to_id("<unk>"),
            "mask": self.custom_tokenizer.token_to_id("<mask>"),
        }
        missing = [name for name, token_id in special_ids.items() if token_id is None]
        if missing:
            raise ValueError(f"Missing special token ids in tokenizer: {', '.join(missing)}")
        return special_ids

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, num_beams: int = 4) -> str:
        if not text or not text.strip():
            return ""
        return self._summarize_scratch(text, max_length)

    def _summarize_scratch(self, text, max_len=128):
        try:
            src_enc = self.custom_tokenizer.encode(text).ids[:max_len]
            if not src_enc:
                return ""

            src = torch.tensor(src_enc, dtype=torch.long).unsqueeze(1).to(self.device)
            tgt_indices = [self.special_ids["sos"]]
            blocked_tokens = {
                self.special_ids["sos"],
                self.special_ids["pad"],
                self.special_ids["unk"],
                self.special_ids["mask"],
            }

            for _ in range(max_len):
                tgt = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1).to(self.device)

                with torch.no_grad():
                    output = self.model(src, tgt)

                logits = output[-1, 0, :].clone()
                for token_id in blocked_tokens:
                    logits[token_id] = -float("inf")

                for token_id in set(tgt_indices[-8:]):
                    if logits[token_id] > 0:
                        logits[token_id] /= 1.5
                    else:
                        logits[token_id] *= 1.5

                if len(tgt_indices) < 5:
                    logits[self.special_ids["eos"]] = -float("inf")

                if len(tgt_indices) >= 4:
                    last_bigram = tuple(tgt_indices[-2:])
                    for index in range(len(tgt_indices) - 2):
                        if tuple(tgt_indices[index:index + 2]) == last_bigram:
                            logits[tgt_indices[index + 2]] = -float("inf")

                next_token = logits.argmax().item()
                if len(tgt_indices) > 2 and next_token == tgt_indices[-1] == tgt_indices[-2]:
                    next_token = logits.topk(2).indices[1].item()

                tgt_indices.append(next_token)
                if next_token == self.special_ids["eos"]:
                    break

            return self.custom_tokenizer.decode(tgt_indices, skip_special_tokens=True).strip()
        except Exception as exc:
            logger.error(f"Error during scratch summarization: {exc}")
            return "Error during generation"


if __name__ == "__main__":
    summarizer = FrenchSummarizer(model_type="scratch", model_path="models/transformer_scratch.pth")
    sample_text = (
        "L'intelligence artificielle est un domaine informatique qui permet de concevoir "
        "des systemes capables d'analyser des donnees, d'apprendre et d'aider a la decision."
    )

    print("\n--- Test Model From Scratch ---")
    print(f"Original Text: {sample_text}")
    print(f"Generated Summary: {summarizer.summarize(sample_text)}")
