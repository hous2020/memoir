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
python src/train_scratch.py --dataset-name xlsum --max-samples 5000
python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 20
```

Datasets disponibles:

```text
xlsum  -> csebuetnlp/xlsum, donnees francaises
mlsum  -> reciTAL/mlsum, donnees francaises
all    -> combine xlsum et mlsum pour l'entrainement
```

Lance l'application locale apres entrainement:

```bash
streamlit run src/app.py
```

## Notes

Le modele doit etre reentraine apres chaque changement important du pipeline d'entrainement. Les poids generes seront sauvegardes dans `models/transformer_scratch.pth`.
