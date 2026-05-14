# Memoir - Resume automatique en francais

Projet de resume automatique de textes en francais avec un modele Transformer entraine depuis zero.

Les donnees sont chargees directement depuis Hugging Face pour eviter de versionner des fichiers lourds.

## Structure

```text
src/
  app.py                 # Interface Streamlit pour le modele scratch
  data_loader.py         # Chargement des datasets depuis Hugging Face Parquet
  evaluation.py          # Evaluation ROUGE du modele scratch
  inspect_data.py        # Inspection rapide des datasets locaux
  summarizer.py          # Inference avec le modele scratch
  train_scratch.py       # Entrainement du Transformer scratch
  train_tokenizer.py     # Entrainement du tokenizer BPE
  transformer_model.py   # Architecture Transformer PyTorch
```

Les dossiers `data/` et `models/` ne sont pas versionnes car ils peuvent etre volumineux.

## Utilisation sur Google Colab

Active d'abord le GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Installe les dependances:

```bash
pip install -r requirements.txt
```

Entraine le tokenizer:

```bash
python src/train_tokenizer.py --dataset-name xlsum
```

Entraine le modele scratch:

```bash
python src/train_scratch.py --dataset-name xlsum
```

Evalue le modele scratch:

```bash
python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 20
```

Pour un test rapide sur Colab:

```bash
python src/train_tokenizer.py --dataset-name xlsum --max-samples 5000
python src/train_scratch.py --dataset-name xlsum --max-samples 5000 --epochs 3
python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 20
```

Pour obtenir de meilleurs resultats, utilise plus d'exemples et plus d'epoques:

```bash
python src/train_tokenizer.py --dataset-name xlsum --max-samples 20000
python src/train_scratch.py --dataset-name xlsum --max-samples 20000 --epochs 8 --batch-size 32
python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 100
```

Pour un entrainement plus ambitieux:

```bash
python src/train_tokenizer.py --dataset-name all --max-samples 50000
python src/train_scratch.py --dataset-name all --max-samples 50000 --epochs 10 --batch-size 32 --d-model 384 --nhead 6 --num-layers 4
python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 100
```

Datasets disponibles:

```text
xlsum   -> csebuetnlp/xlsum, donnees francaises (~20k exemples)
mlsum   -> reciTAL/mlsum, donnees francaises (~200k exemples)
lemonde -> trivago/mlsum, donnees francaises supplementaires (~30k exemples)
all     -> combine xlsum, mlsum et lemonde pour l'entrainement (~250k exemples)
```

Lance l'application locale apres entrainement:

```bash
streamlit run src/app.py
```

## Ameliorations recentes

### Entraînement optimisé
- **Gradient clipping** pour stabiliser l'entraînement (valeur par défaut: 1.0)
- **Learning rate scheduler** CosineAnnealingLR pour une meilleure convergence
- **Validation loop** avec calcul des métriques ROUGE en temps réel
- **Early stopping** avec patience de 3 epochs pour éviter le surapprentissage
- **Sauvegarde automatique** du meilleur modèle selon le validation loss
- **Barre de progression** avec affichage du loss en temps réel

### Datasets augmentés
- **XLSUM**: 2 fichiers d'entraînement (au lieu de 1)
- **MLSUM**: 5 fichiers d'entraînement (au lieu de 3)
- **LEMONDE**: Nouveau dataset ajouté
- **ALL**: Combine les 3 sources pour ~250k exemples disponibles

## Notes

Le modele doit etre reentraine apres chaque changement important du pipeline d'entrainement. Les poids generes seront sauvegardes dans `models/transformer_scratch.pth` et la configuration dans `models/transformer_scratch_config.json`.
