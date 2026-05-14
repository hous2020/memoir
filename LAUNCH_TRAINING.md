# ============================================================================
# 🚀 ENTRAÎNEMENT COMPLET - GUIDE D'EXÉCUTION SUR COLAB
# ============================================================================
# 
# TEMPS ESTIMÉ : 8-12 heures sur GPU Tesla T4
# DATASETS : XLSum + MLSum (~210k exemples)
# ARCHITECTURE : d_model=512, 6 couches transformer, 8 heads
#
# INSTRUCTIONS :
# 1. Ouvrez le notebook colab_training.ipynb sur Google Colab
# 2. Activez le GPU : Execution > Modifier le type d'exécution > GPU
# 3. Exécutez les cellules dans l'ordre
# ============================================================================

# ----------------------------------------------------------------------------
# CELLULE 1 : Vérification du GPU (1 seconde)
# ----------------------------------------------------------------------------
import torch

print('✅ CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('🎮 GPU:', torch.cuda.get_device_name(0))
    print('📊 Mémoire GPU:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'Go')

# ----------------------------------------------------------------------------
# CELLULE 2 : Clone ou mise à jour du repository (10-30 secondes)
# ----------------------------------------------------------------------------
import os

if os.path.exists('/content/memoir/.git'):
    %cd /content/memoir
    !git pull
    print("✅ Repository mis à jour")
else:
    %cd /content
    !git clone https://github.com/hous2020/memoir.git
    %cd /content/memoir
    print("✅ Repository cloné")

# ----------------------------------------------------------------------------
# CELLULE 3 : Installation des dépendances (1-2 minutes)
# ----------------------------------------------------------------------------
!pip install -r requirements.txt -q
print("✅ Dépendances installées")

# ----------------------------------------------------------------------------
# CELLULE 4 : ENTRAÎNEMENT COMPLET (8-12 heures)
# ----------------------------------------------------------------------------
# ⚠️ CETTE CELLULE VA S'EXÉCUTER TOUTE SEULE PENDANT PLUSIEURS HEURES
# Ne fermez pas l'onglet Colab !
# 
# Ce script exécute automatiquement :
#   Étape 1/3 : Entraînement du tokenizer (10-15 min)
#   Étape 2/3 : Entraînement du modèle (8-11 heures)
#   Étape 3/3 : Évaluation avec métriques ROUGE (5-10 min)
# ----------------------------------------------------------------------------

print("🚀 Démarrage de l'entraînement complet...")
print("⏱️  Temps estimé : 8-12 heures")
print("📊 Datasets : XLSum + MLSum (100k exemples)")
print("🏗️  Architecture : d_model=512, 6 couches, 8 heads")
print("📈 Epochs : 15 avec validation et early stopping")
print("="*70)

!python src/train_complete.py

# ----------------------------------------------------------------------------
# APRÈS L'ENTRAÎNEMENT : Sauvegarde dans Google Drive (optionnel)
# ----------------------------------------------------------------------------
# Décommentez ces lignes pour sauvegarder votre modèle :

# from google.colab import drive
# drive.mount('/content/drive')
# 
# !mkdir -p /content/drive/MyDrive/memoir_models
# !cp models/transformer_scratch.pth /content/drive/MyDrive/memoir_models/
# !cp data/custom_tokenizer.json /content/drive/MyDrive/memoir_models/
# !cp models/transformer_scratch_config.json /content/drive/MyDrive/memoir_models/
# 
# print("✅ Modèle sauvegardé dans Google Drive !")

# ----------------------------------------------------------------------------
# TEST RAPIDE APRÈS ENTRAÎNEMENT
# ----------------------------------------------------------------------------
# Pour tester votre modèle entraîné :

# !python src/evaluation.py --model-type scratch --dataset-name xlsum --num-samples 50

# ----------------------------------------------------------------------------
# 💡 CONSEILS :
# ----------------------------------------------------------------------------
# 1. Laissez Colab tourner toute la nuit
# 2. Vous pouvez suivre la progression via les barres tqdm
# 3. Les métriques ROUGE s'affichent après chaque epoch
# 4. Le meilleur modèle est automatiquement sauvegardé
# 5. Si la session expire, recommencez depuis la cellule 2 (le code est déjà là)
# ----------------------------------------------------------------------------
