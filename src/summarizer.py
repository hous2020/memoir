import torch
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrenchSummarizer:
    def __init__(self, model_name: str = "moussaKam/barthez-orangesum-abstract"):
        """
        Initializes the French Summarizer with a pre-trained model.
        Args:
            model_name: The Hugging Face model hub name.
        """
        logger.info(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, num_beams: int = 4) -> str:
        """
        Summarizes the given text.
        Args:
            text: The input text to summarize.
            max_length: Maximum length of the summary.
            min_length: Minimum length of the summary.
            num_beams: Number of beams for beam search.
        Returns:
            The generated summary string.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for summarization.")
            return ""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                [text], 
                max_length=1024, 
                truncation=True, 
                return_tensors="pt",
                padding="longest"
            )
            
            # Move inputs to device
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            # Generate summary
            summary_ids = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                num_beams=num_beams, 
                max_length=max_length, 
                min_length=min_length,
                early_stopping=True
            )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise

if __name__ == "__main__":
    # Test the summarizer
    try:
        summarizer = FrenchSummarizer()
        
        sample_text = (
            "L'intelligence artificielle (IA) est un processus d'imitation de l'intelligence humaine qui repose sur la création "
            "et l'application d'algorithmes exécutés dans un environnement informatique dynamique. Son but est de permettre "
            "à des ordinateurs de penser et d'agir comme des êtres humains. Pour y parvenir, trois composants sont nécessaires : "
            "des systèmes informatiques, des données avec des systèmes de gestion, et des algorithmes d'IA avancés (code). "
            "Pour se rapprocher le plus possible du comportement humain, l'intelligence artificielle a besoin d'une quantité "
            "de données et d'une capacité de traitement élevées."
        )
        
        print("\nOriginal Text:")
        print(sample_text)
        
        print("\nGenerating Summary...")
        summary = summarizer.summarize(sample_text)
        
        print("\nSummary:")
        print(summary)
        
    except Exception as e:
        print(f"Test failed: {e}")
