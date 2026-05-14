#!/usr/bin/env python3
"""
Script d'entraînement complet sur TOUS les datasets
À exécuter depuis le notebook Colab pour un entraînement maximal
"""

import subprocess
import sys

def run_command(command, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Commande: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Erreur lors de: {description}")
        sys.exit(1)
    
    print(f"\n✅ {description} - TERMINÉ")

def main():
    print("\n" + "="*70)
    print("🎯 ENTRAÎNEMENT MAXIMAL - TOUS LES DATASETS")
    print("="*70)
    print("\nCe script va:")
    print("1. Entraîner le tokenizer sur 100k exemples (XLSum + MLSum + LeMonde)")
    print("2. Entraîner le modèle avec architecture optimale (15 epochs)")
    print("3. Évaluer le modèle sur 200 exemples de test")
    print("\n⏱️  Temps estimé: 8-12 heures sur GPU")
    print("="*70)
    
    # Étape 1: Entraînement du tokenizer
    run_command(
        "python src/train_tokenizer.py --dataset-name all --max-samples 100000 --vocab-size 40000",
        "Étape 1/3: Entraînement du tokenizer sur tous les datasets"
    )
    
    # Étape 2: Entraînement du modèle
    run_command(
        "python src/train_scratch.py --dataset-name all --max-samples 100000 --epochs 15 --batch-size 32 --d-model 512 --nhead 8 --num-layers 6 --learning-rate 0.00005 --max-seq-len 150",
        "Étape 2/3: Entraînement du modèle (architecture optimale)"
    )
    
    # Étape 3: Évaluation
    run_command(
        "python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 200",
        "Étape 3/3: Évaluation du modèle"
    )
    
    print("\n" + "="*70)
    print("🎉 ENTRAÎNEMENT COMPLET TERMINÉ !")
    print("="*70)
    print("\n📊 Résultats:")
    print("   - Tokenizer: data/custom_tokenizer.json (vocabulaire: 40000)")
    print("   - Modèle: models/transformer_scratch.pth")
    print("   - Configuration: models/transformer_scratch_config.json")
    print("\n💡 Vous pouvez maintenant:")
    print("   - Tester l'application: streamlit run src/app.py")
    print("   - Sauvegarder dans Google Drive (voir cellule dédiée)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
