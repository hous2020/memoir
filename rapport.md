# Rapport Technique — Modèle de Résumé Automatique From Scratch (Français)

## 1. Objectif et Portée
- Concevoir et entraîner un modèle de résumé automatique de textes longs en français, sans recourir à des modèles pré-entraînés.
- Générer des résumés abstraits (abstractive summarization) informatifs, cohérents et grammaticalement corrects.
- Portée: articles de presse, rapports techniques, documents informatifs; longueur d’entrée ciblée jusqu’à ~1 024 tokens.

## 2. Données: Collecte et Préparation
- Sources candidates:
  - MLSUM (FR), XL-Sum (FR), WikiSum (FR), jeux d’articles francophones (presse, blogs).
  - Corpus interne: documents PDF/TXT convertis, avec résumés rédigés manuellement.
- Étiquettes attendues: champs “text” (document), “summary” (résumé de référence).
- Nettoyage:
  - Normalisation Unicode, suppression des artefacts (HTML, balises), unification des guillemets.
  - Filtrage longueur: conserver text ∈ [200, 3 000] tokens; summary ∈ [30, 300] tokens.
  - Détection du français (langid) et retrait des lignes non FR.
- Partition:
  - Entraînement 80%, validation 10%, test 10% (stratifier par domaine si possible).
- Tokenisation à entraîner:
  - SentencePiece (BPE) ou Unigram, vocabulaire cible: 32k–50k.
  - Pré-tokenisation: lowercasing facultatif; conserver accents; normalizer NFC.
  - Sauvegarde: tokenizer.model + vocab.json pour reproductibilité.

## 3. Conception du Modèle
- Choix: Transformer séquence-à-séquence (Encoder–Decoder) from scratch.
- Justification:
  - Excellentes performances sur les tâches de génération + meilleure parallélisation que RNN.
  - Capacité à capturer dépendances longues via attention multi-tête.
- Paramètres initiaux:
  - Vocabulaire: 32k
  - Embedding: 512
  - Couches Encoder: 6; Decoder: 6
  - Têtes d’attention: 8; d_model: 512; d_ff: 2 048
  - Activations: GELU; Normalisation: LayerNorm; Dropout: 0.1–0.3
  - Positionnel: encodage sinusoïdal ou appris
  - Tying: lier embeddings entrée/sortie pour réduire les paramètres
  - Masquage: causal mask côté décodeur; padding mask pour attention
- Spécificités pour textes longs:
  - Troncature progressive; windowed attention ou hiérarchique si >1 024 tokens (itération future).
  - Option pointer-generator (copie) à considérer si forte présence de noms propres.

## 4. Implémentation (PyTorch)
- Framework: PyTorch (modules autonomes).
- Composants:
  - MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer, PositionalEncoding.
  - Classe Seq2Seq: forward(src, tgt) avec teacher forcing en entraînement.
- Perte:
  - Cross‑entropy avec label smoothing (ε ≈ 0.1), ignorer PAD via `ignore_index`.
- Optimiseur:
  - AdamW (β1=0.9, β2=0.98, weight_decay=0.01).
- Scheduler:
  - Warmup + décroissance Cosine ou Noam (learning rate proportionnel à d_model⁻⁰·⁵).
- Bonnes pratiques:
  - Gradient clipping (norm ≤ 1.0).
  - Mixed precision (fp16) via autocast pour accélérer.
  - Seed fixe et cudnn deterministic pour reproductibilité.

## 5. Pipeline d’Entraînement
- Gestion des données:
  - DataLoader avec batch dynamique (bucketing par longueur).
  - Padding + masks; tokenisation offline pour accélérer IO.
- Callbacks:
  - Early stopping sur ROUGE-L validation (patience 5–8).
  - Checkpointing (meilleur modèle + derniers N), sauvegarde état optimiseur/scheduler.
  - Logging: TensorBoard + fichiers .csv des métriques par epoch.
- Boucle:
  - Forward pass encoder sur src; décodeur conditionné sur `tgt_in` (décalé).
  - Perte sur `tgt_out` (décalé), backprop, step optimiseur + scheduler.
  - Évaluation périodique toutes K étapes sur un sous-ensemble validation.

## 6. Évaluation
- Métriques:
  - ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum (F1 et/ou recall).
  - Longueur du résumé, taux de compression, répétitions n‑grammes (éviter le “looping”).
  - Perplexité (indicative) pour stabiliser l’entraînement.
- Protocole:
  - Ensemble de test disjoint, diversité de domaines; rapporter moyenne + écart type.
  - Comparaison avec baselines simples (lead‑k, extraction TF‑IDF).
- Cibles de performance (indicatives pour un modèle de base):
  - ROUGE‑1 ≥ 35, ROUGE‑2 ≥ 15, ROUGE‑L ≥ 30 sur corpus journalistique FR.

## 7. Itérations et Hyperparamètres
- Recherche:
  - Grid/Random sur d_model {512, 768}, couches {4, 6, 8}, heads {8, 12}, dropout {0.1, 0.2, 0.3}.
  - Longueur max entrée {768, 1 024, 1 536}; tie embeddings {on/off}.
  - Schedulers {Noam, Cosine}; label smoothing {0.0, 0.1}.
- Ablations:
  - Retrait pointer, changement activation (ReLU vs GELU), LayerNorm placement (pre‑LN vs post‑LN).
- Critères d’arrêt:
  - Stabilisation ROUGE validation; convergence perte; éviter sur‑apprentissage (gap train/val).

## 8. Documentation et Traçabilité
- Scripts et artefacts:
  - Tokenizer entraîné (sentencepiece.model) avec config du corpus.
  - Config YAML des hyperparams par expérimentation.
  - Journaux d’entraînement (TensorBoard), checkpoints datés et versionnés.
- Reproductibilité:
  - Seeds, versions exactes des dépendances (requirements.txt).
  - Description des splits et des règles de nettoyage.

## 9. Déploiement et Intégration
- Export:
  - Poids PyTorch (.pt/.bin); option TorchScript pour inference.
  - Wrapper d’inférence préparé (pré/post‑traitement).
- Intégration UI:
  - Application Streamlit existante: remplacement du fournisseur de résumés par le modèle maison.
  - Points d’extension dans:
    - [summarizer.py](file:///c:/Users/HOUESSOU/Desktop/projet/memoir/src/summarizer.py)
    - [app.py](file:///c:/Users/HOUESSOU/Desktop/projet/memoir/src/app.py)
    - [evaluation.py](file:///c:/Users/HOUESSOU/Desktop/projet/memoir/src/evaluation.py)
    - [data_loader.py](file:///c:/Users/HOUESSOU/Desktop/projet/memoir/src/data_loader.py)

## 10. Risques et Atténuations
- Données insuffisantes:
  - Enrichir corpus, augmentation de données (paraphrase contrôlée), nettoyage rigoureux.
- Coût calcul:
  - Démarrer avec petite config (6×6, d_model=512), entraînement fp16; gradient accumulation.
- Hallucinations:
  - Penaliser répétitions via coverage loss; contraintes de longueur; validation humaine.
- Biais linguistiques:
  - Diversifier sources, audits; mettre en place guidelines d’usage responsable.

## 11. Plan de Travail (Indicatif)
- S1: collecte/cleanup + entraînement tokenizer
- S2–S3: baseline Transformer (6×6, 512), premiers résultats
- S4: itérations hyperparams, ablations, coverage
- S5: stabilisation, packaging inference, intégration UI, rédaction finale

## 12. Critères de Validation
- Fonctionnalité: génère un résumé pour ≥ 95% des entrées sans erreur.
- Qualité: dépasse baselines extractives; atteint seuils ROUGE cibles.
- Robustesse: pas de crash sur textes longs; contrôle des répétitions.
- Documentation: toutes les décisions techniques tracées; scripts reproductibles.

## 13. Annexes — Spécifications d’Implémentation
- Entraîner le tokenizer (exemple):
  - Corpus concaténé en texte brut; SentencePiece BPE vocab=32 000; normalizer NFC.
- Format d’entrée du modèle:
  - `src_ids` (pad à gauche), `tgt_in` et `tgt_out` (décalage d’un token), masks booléens.
- Inference:
  - Beam search (beam=4–8), longueur min/max; pénalité de répétition n‑gram (option).
- Journalisation:
  - Perte, ROUGE val/test par epoch; temps moyen par batch; utilisation GPU.

## 14. Conclusion
Ce rapport définit un plan détaillé pour concevoir, implémenter et évaluer un modèle de résumé automatique en français entièrement from scratch. L’architecture Transformer séquence‑à‑séquence, associée à une pipeline de données robuste et un protocole d’entraînement rigoureux, permet d’atteindre des performances satisfaisantes sans dépendre de modèles pré‑entraînés. Les sections fournissent les choix, hyperparamètres, métriques, risques et étapes nécessaires pour un livrable industriel et scientifique.

