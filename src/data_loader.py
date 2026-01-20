from datasets import load_dataset

def load_french_summarization_data(dataset_name="csebuetnlp/xlsum", split="test", subset="french"):
    """
    Loads a French summarization dataset.
    Args:
        dataset_name: Name of the dataset.
        split: 'train', 'validation', or 'test'.
        subset: specific configuration for the dataset.
    Returns:
        A dataset object.
    """
    print(f"Loading {dataset_name} ({subset}) - {split} split...")
    try:
        # xlsum usually works without trust_remote_code or with it, but let's try standard load
        dataset = load_dataset(dataset_name, subset, split=split)
            
        print(f"Successfully loaded {len(dataset)} examples.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to dummy data if online loading fails
        print("Falling back to dummy data for demonstration.")
        from datasets import Dataset
        dummy_data = {
            "text": [
                "L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des machines capables de simuler l'intelligence humaine. "
                "Elle englobe diverses techniques, dont l'apprentissage automatique et le traitement du langage naturel. "
                "Les modèles de langage comme GPT ou BERT ont révolutionné la façon dont les ordinateurs comprennent et génèrent du texte. "
                "Ces avancées permettent des applications variées, allant de la traduction automatique à la génération de code."
            ],
            "summary": [
                "L'IA simule l'intelligence humaine via des techniques comme l'apprentissage automatique. Les modèles de langage récents ont transformé le traitement du texte."
            ]
        }
        return Dataset.from_dict(dummy_data)

if __name__ == "__main__":
    # Test loading a small sample
    ds = load_french_summarization_data(dataset_name="csebuetnlp/xlsum", split="test[:10]", subset="french")
    if ds:
        print("\nExample entry:")
        print(f"Text: {ds[0]['text'][:200]}...")
        print(f"Summary: {ds[0]['summary']}")
