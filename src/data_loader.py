import os

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk


# Datasets français de résumé chargés via fichiers Parquet (évite les loading scripts bloqués).
# Format du chemin parquet : hf://datasets/{repo}@refs/convert/parquet/{config}/{split}/{shard}.parquet
HF_DATASETS = {
    "xlsum": {
        "repo_id": "csebuetnlp/xlsum",
        "text_column": "text",
        "summary_column": "summary",
        # ~38k train — articles BBC en français
        "parquet_files": {
            "train":      ["french/train/0000.parquet"],
            "test":       ["french/test/0000.parquet"],
            "validation": ["french/validation/0000.parquet"],
        },
    },
    "mlsum": {
        "repo_id": "reciTAL/mlsum",
        "text_column": "text",
        "summary_column": "summary",
        # ~33k train — articles de presse français
        "parquet_files": {
            "train":      ["fr/train/0000.parquet", "fr/train/0001.parquet", "fr/train/0002.parquet"],
            "test":       ["fr/test/0000.parquet"],
            "validation": ["fr/validation/0000.parquet"],
        },
    },
    "wiki_lingua_fr": {
        "repo_id": "GEM/wiki_lingua",
        "text_column": "source",
        "summary_column": "target",
        # ~41k train — articles Wikipedia FR avec résumé
        "parquet_files": {
            "train":      ["fr/train/0000.parquet"],
            "test":       ["fr/test/0000.parquet"],
            "validation": ["fr/validation/0000.parquet"],
        },
    },
}

ALL_DATASET_NAMES = list(HF_DATASETS.keys())


def _normalize_dataset(dataset, text_column="text", summary_column="summary"):
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if summary_column != "summary":
        dataset = dataset.rename_column(summary_column, "summary")

    keep_columns = {"text", "summary"}
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    return dataset.filter(lambda item: bool(item["text"]) and bool(item["summary"]))


def _select_max_samples(dataset, max_samples=None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_huggingface_dataset(dataset_name="xlsum", split="train", max_samples=None):
    """
    Charge un dataset de résumé depuis Hugging Face via fichiers Parquet.

    Args:
        dataset_name: "xlsum", "mlsum", "wiki_lingua_fr", ou "all".
        split: "train", "test" ou "validation".
        max_samples: Limite optionnelle (utile pour tests rapides sur Colab).
                     En mode "all", la limite est répartie équitablement entre datasets.
    """
    if dataset_name == "all":
        print(f"Chargement de tous les datasets ({len(ALL_DATASET_NAMES)} sources)...")
        num_sources = len(ALL_DATASET_NAMES)
        # Répartir max_samples équitablement pour éviter de saturer la mémoire
        per_dataset = (max_samples // num_sources) if max_samples else None

        datasets = []
        for name in ALL_DATASET_NAMES:
            try:
                ds = load_huggingface_dataset(name, split=split, max_samples=per_dataset)
                datasets.append(ds)
                print(f"  + {name}: {len(ds)} exemples")
            except Exception as exc:
                print(f"  ! Impossible de charger {name}: {exc}")

        if not datasets:
            raise ValueError("Aucun dataset n'a pu être chargé.")

        combined = concatenate_datasets(datasets)
        print(f"Dataset combiné: {len(combined)} exemples depuis {len(datasets)} sources.")
        return combined

    if dataset_name not in HF_DATASETS:
        available = ", ".join([*ALL_DATASET_NAMES, "all"])
        raise ValueError(f"Dataset inconnu '{dataset_name}'. Disponibles: {available}")

    spec = HF_DATASETS[dataset_name]
    if split not in spec["parquet_files"]:
        available_splits = ", ".join(spec["parquet_files"].keys())
        raise ValueError(f"Split '{split}' inconnu pour {dataset_name}. Disponibles: {available_splits}")

    data_files = [
        f"hf://datasets/{spec['repo_id']}@refs/convert/parquet/{filename}"
        for filename in spec["parquet_files"][split]
    ]

    print(f"Chargement de {dataset_name} (split={split})...")
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = _normalize_dataset(dataset, spec["text_column"], spec["summary_column"])
    dataset = _select_max_samples(dataset, max_samples)
    print(f"Chargé: {len(dataset)} exemples depuis {dataset_name}.")
    return dataset


def load_local_dataset(path):
    """Charge un dataset Hugging Face sauvegardé localement."""
    if os.path.exists(path):
        return load_from_disk(path)
    print(f"Dataset introuvable à {path}.")
    return None


def load_all_combined_datasets(base_dir="data"):
    """Chargeur local legacy. Préférer load_huggingface_dataset pour Colab."""
    all_train = []
    print(f"Recherche de datasets locaux dans {base_dir}...")

    if not os.path.exists(base_dir):
        print(f"Répertoire local introuvable: {base_dir}")
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
                print(f"Chargement local de {item}...")
                dataset = load_from_disk(item_path)
                if isinstance(dataset, DatasetDict):
                    if "train" in dataset:
                        all_train.append(_normalize_dataset(dataset["train"]))
                else:
                    all_train.append(_normalize_dataset(dataset))
        except Exception as exc:
            print(f"Impossible de charger {item}: {exc}")

    if not all_train:
        print("Aucun dataset local trouvé.")
        return None

    combined_train = concatenate_datasets(all_train)
    print(f"Dataset local combiné: {len(combined_train)} exemples.")
    return combined_train


if __name__ == "__main__":
    for name in ALL_DATASET_NAMES:
        print(f"\n=== {name} ===")
        ds = load_huggingface_dataset(dataset_name=name, split="train", max_samples=3)
        print(ds)
        print(ds[0])
