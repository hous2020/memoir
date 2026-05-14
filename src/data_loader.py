import os

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


HF_DATASETS = {
    "xlsum": {
        "repo_id": "csebuetnlp/xlsum",
        "text_column": "text",
        "summary_column": "summary",
        "parquet_files": {
            "train": ["french/train/0000.parquet"],
            "test": ["french/test/0000.parquet"],
            "validation": ["french/validation/0000.parquet"],
        },
    },
    "mlsum": {
        "repo_id": "reciTAL/mlsum",
        "text_column": "text",
        "summary_column": "summary",
        "parquet_files": {
            "train": ["fr/train/0000.parquet", "fr/train/0001.parquet", "fr/train/0002.parquet"],
            "test": ["fr/test/0000.parquet"],
            "validation": ["fr/validation/0000.parquet"],
        },
    },
}


def _normalize_dataset(dataset, text_column="text", summary_column="summary"):
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if summary_column != "summary":
        dataset = dataset.rename_column(summary_column, "summary")

    keep_columns = {"text", "summary"}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    return dataset.filter(lambda item: bool(item["text"]) and bool(item["summary"]))


def _select_max_samples(dataset, max_samples=None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_huggingface_dataset(dataset_name="xlsum", split="train", max_samples=None):
    """
    Loads a summarization dataset directly from Hugging Face.

    Args:
        dataset_name: "xlsum", "mlsum", "lemonde", or "all".
        split: Hugging Face split to load, for example "train", "test", or "validation".
        max_samples: Optional limit useful for fast Colab experiments.
    """
    if dataset_name == "all":
        print("Loading all datasets combined...")
        datasets = []
        for name in HF_DATASETS:
            try:
                ds = load_huggingface_dataset(name, split=split, max_samples=max_samples)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded.")
        
        combined = concatenate_datasets(datasets)
        print(f"Combined dataset: {len(combined)} examples from {len(datasets)} sources.")
        
        if max_samples:
            combined = combined.select(range(min(max_samples, len(combined))))
        
        return combined

    if dataset_name not in HF_DATASETS:
        available = ", ".join([*HF_DATASETS.keys(), "all"])
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    spec = HF_DATASETS[dataset_name]
    if split not in spec["parquet_files"]:
        available_splits = ", ".join(spec["parquet_files"].keys())
        raise ValueError(f"Unknown split '{split}' for {dataset_name}. Available: {available_splits}")

    data_files = [
        f"hf://datasets/{spec['repo_id']}@refs/convert/parquet/{filename}"
        for filename in spec["parquet_files"][split]
    ]

    print(f"Loading {dataset_name} from Hugging Face Parquet files (split={split})...")
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = _normalize_dataset(dataset, spec["text_column"], spec["summary_column"])
    dataset = _select_max_samples(dataset, max_samples)
    print(f"Loaded {len(dataset)} examples from {dataset_name}.")
    return dataset


def load_local_dataset(path):
    """
    Loads a Hugging Face dataset saved on disk.
    """
    if os.path.exists(path):
        return load_from_disk(path)
    print(f"Dataset not found at {path}.")
    return None


def load_all_combined_datasets(base_dir="data"):
    """
    Legacy local loader. Prefer load_huggingface_dataset for Colab/public repos.
    """
    all_train = []
    print(f"Searching local datasets in {base_dir}...")

    if not os.path.exists(base_dir):
        print(f"Local data directory does not exist: {base_dir}")
        return None

    skip_dirs = {"temp_cache", "abstract_tmp", "model_cache"}

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path) or item in skip_dirs:
            continue

        try:
            if os.path.exists(os.path.join(item_path, "dataset_dict.json")) or os.path.exists(
                os.path.join(item_path, "train")
            ):
                print(f"Loading local dataset {item}...")
                dataset = load_from_disk(item_path)
                if isinstance(dataset, DatasetDict):
                    if "train" in dataset:
                        all_train.append(_normalize_dataset(dataset["train"]))
                else:
                    all_train.append(_normalize_dataset(dataset))
        except Exception as exc:
            print(f"Could not load {item}: {exc}")

    if not all_train:
        print("No local training dataset found.")
        return None

    combined_train = concatenate_datasets(all_train)
    print(f"Combined local dataset size: {len(combined_train)} examples.")
    return combined_train


if __name__ == "__main__":
    dataset = load_huggingface_dataset(dataset_name="xlsum", split="train", max_samples=5)
    print(dataset)
