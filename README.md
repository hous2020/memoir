# Memoir - Resume automatique en francais

Projet de resume automatique de textes en francais avec deux approches:

- un modele Transformer entraine from scratch;
- un modele pre-entraine BARThez pour comparaison.

## Structure

```text
src/
  app.py                 # Interface Streamlit
  data_loader.py         # Telechargement et chargement des datasets
  evaluation.py          # Evaluation ROUGE
  inspect_data.py        # Inspection rapide des datasets locaux
  summarizer.py          # Inference avec BARThez ou le modele scratch
  train_scratch.py       # Entrainement du Transformer scratch
  train_tokenizer.py     # Entrainement du tokenizer BPE
  transformer_model.py   # Architecture Transformer PyTorch
```

Les dossiers `data/` et `models/` ne sont pas versionnes car ils peuvent etre volumineux.
Les donnees d'entrainement sont chargees directement depuis Hugging Face.

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

Entraine le tokenizer si `data/custom_tokenizer.json` n'existe pas:

```bash
python src/train_tokenizer.py --dataset-name xlsum
```

Entraine le modele scratch:

```bash
python src/train_scratch.py --dataset-name xlsum
```

Evalue les modeles:

```bash
python src/evaluation.py --model-type both --dataset-name xlsum --num-samples 20
```

Pour un test rapide sur Colab:

```bash
python src/train_tokenizer.py --dataset-name xlsum --max-samples 5000
python src/train_scratch.py --dataset-name xlsum --max-samples 5000
python src/evaluation.py --model-type both --dataset-name xlsum --num-samples 20
```

Datasets disponibles:

```text
xlsum  -> csebuetnlp/xlsum, configuration french
mlsum  -> reciTAL/mlsum, configuration fr
all    -> combine xlsum et mlsum pour l'entrainement
```

Lance l'application locale:

```bash
streamlit run src/app.py
```

## Notes

Le modele scratch doit etre reentraine apres modification du pipeline d'entrainement. Les poids locaux existants peuvent fonctionner techniquement, mais leur qualite depend de l'ancien entrainement.
