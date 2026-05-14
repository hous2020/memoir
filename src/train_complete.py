#!/usr/bin/env python3
"""
Script d'entraînement complet sur TOUS les datasets disponibles.
À exécuter depuis le notebook Colab.

Datasets utilisés (5 sources françaises) :
  - XLSum french      (~38k train)
  - MLSUM fr          (~33k train)
  - OrangeSum abstract (~30k train)
  - OrangeSum title    (~30k train)
  - OrangeSum wikilead (~2.5M train — limité par --max-samples)
"""

import subprocess
import sys


def run_command(command, description):
    print(f"\n{'='*70}")
    print(f">>> {description}")
    print(f"{'='*70}")
    print(f"Commande: {command}\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\nErreur lors de: {description}")
        sys.exit(1)

    print(f"\n[OK] {description} - TERMINE")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement complet sur tous les datasets.")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Limite par dataset (défaut: 100k → 500k total)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    total_samples = args.max_samples * 5  # 5 datasets

    print("\n" + "=" * 70)
    print("ENTRAINEMENT MAXIMAL - TOUS LES DATASETS FRANCAIS")
    print("=" * 70)
    print(f"\nDatasets: xlsum | mlsum | orangesum_abstract | orangesum_title | orangesum_wikilead")
    print(f"Limite: {args.max_samples:,} exemples/dataset = ~{total_samples:,} total")
    print(f"Architecture: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("=" * 70)

    run_command(
        f"python src/train_tokenizer.py"
        f" --dataset-name all"
        f" --max-samples {args.max_samples}"
        f" --vocab-size {args.vocab_size}",
        "Etape 1/3: Tokenizer BPE sur tous les datasets",
    )

    run_command(
        f"python src/train_scratch.py"
        f" --dataset-name all"
        f" --max-samples {args.max_samples}"
        f" --epochs {args.epochs}"
        f" --batch-size {args.batch_size}"
        f" --d-model {args.d_model}"
        f" --nhead {args.nhead}"
        f" --num-layers {args.num_layers}"
        f" --learning-rate 0.0001"
        f" --max-seq-len 128",
        "Etape 2/3: Entraînement du modèle Transformer",
    )

    run_command(
        "python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 200",
        "Etape 3/3: Evaluation ROUGE sur XLSum test",
    )

    print("\n" + "=" * 70)
    print("ENTRAINEMENT COMPLET TERMINE !")
    print("=" * 70)
    print("\nFichiers produits:")
    print(f"  - Tokenizer : data/custom_tokenizer.json (vocab: {args.vocab_size:,})")
    print(f"  - Modele    : models/transformer_scratch.pth")
    print(f"  - Config    : models/transformer_scratch_config.json")
    print("\nPour tester: streamlit run src/app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
