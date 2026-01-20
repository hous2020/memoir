import evaluate
from data_loader import load_french_summarization_data
from summarizer import FrenchSummarizer
from tqdm import tqdm

def evaluate_model(num_samples=5):
    """
    Evaluates the summarization model using ROUGE metrics.
    Args:
        num_samples: Number of samples to evaluate on.
    """
    # Load data
    dataset = load_french_summarization_data(dataset_name="csebuetnlp/xlsum", split=f"test[:{num_samples}]", subset="french")
    if dataset is None:
        print("Could not load dataset for evaluation.")
        return

    # Load model
    summarizer = FrenchSummarizer()
    
    # Load metric
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    
    print(f"Generating summaries for {len(dataset)} samples...")
    for item in tqdm(dataset):
        text = item["text"]
        reference_summary = item["summary"]
        
        # Generate summary
        generated_summary = summarizer.summarize(text)
        
        predictions.append(generated_summary)
        references.append(reference_summary)
        
    # Compute metrics
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\nEvaluation Results (ROUGE):")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
        
    # Show one example
    print("\n--- Example ---")
    print(f"Original Summary: {references[0]}")
    print(f"Generated Summary: {predictions[0]}")

if __name__ == "__main__":
    evaluate_model(num_samples=3)
