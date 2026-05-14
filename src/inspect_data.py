from datasets import load_from_disk
import os

def inspect_local_dataset(path="data/mlsum_local/train"):
    """
    Lit et affiche le contenu d'un dataset Hugging Face sauvegardé localement.
    """
    if not os.path.exists(path):
        print(f"Le dossier {path} n'existe pas. Veuillez d'abord charger les données.")
        return

    try:
        print(f"Lecture du dataset local : {path}...")
        dataset = load_from_disk(path)
        
        print(f"\nNombre d'exemples trouvés : {len(dataset)}")
        print("-" * 50)
        
        for i, example in enumerate(dataset):
            print(f"\nExemple {i+1}:")
            print(f"TEXTE COMPLET :\n{example['text'][:500]}...") # On limite l'affichage pour la lisibilité
            print(f"\nRÉSUMÉ :\n{example['summary']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")

if __name__ == "__main__":
    inspect_local_dataset()
