import streamlit as st
import torch
from summarizer import FrenchSummarizer
import logging

# Configuration de la page (doit être la première commande Streamlit)
st.set_page_config(
    page_title="Résumé Automatique - BARThez",
    page_icon="📝",
    layout="wide",  # Utilisation de toute la largeur de l'écran
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Fonctions utilitaires ---
def count_words(text):
    if not text:
        return 0
    return len(text.split())

@st.cache_resource
def load_summarizer():
    return FrenchSummarizer()

# --- Interface Utilisateur ---

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    st.divider()
    
    model_params = st.expander("Configuration du Modèle", expanded=True)
    with model_params:
        min_length = st.slider("Longueur minimale", 10, 100, 40, help="Le résumé ne fera pas moins que ce nombre de tokens.")
        max_length = st.slider("Longueur maximale", 50, 400, 150, help="Le résumé ne dépassera pas ce nombre de tokens.")
        num_beams = st.slider("Faisceaux (Beams)", 1, 8, 4, help="Qualité de la recherche. Plus élevé = meilleur mais plus lent.")
    
    st.info("💡 Astuce : Ajustez la longueur maximale en fonction de la taille de votre texte source.")
    st.markdown("---")
    st.caption("v1.1 - Propulsé par BARThez")

# En-tête principal
st.title("📝 Résumé Automatique de Documents")
st.markdown("Générez des synthèses précises de vos textes longs en quelques secondes grâce à l'IA.")

# Chargement du modèle
try:
    with st.spinner("Initialisation du moteur d'IA..."):
        summarizer = load_summarizer()
except Exception as e:
    st.error(f"Erreur critique lors du chargement du modèle : {e}")
    st.stop()

# Création de deux colonnes principales
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Document Source")
    
    # Onglets pour choisir la source (Texte ou Fichier)
    tab_text, tab_file = st.tabs(["Saisie Manuelle", "Importer un Fichier"])
    
    input_text = ""
    
    with tab_text:
        text_area_val = st.text_area(
            "Collez votre texte ici", 
            height=400, 
            placeholder="Copiez-collez ici le contenu d'un article, d'un rapport...",
            key="input_text_area"
        )
        if text_area_val:
            input_text = text_area_val

    with tab_file:
        uploaded_file = st.file_uploader("Choisissez un fichier texte (.txt, .md)", type=["txt", "md"])
        if uploaded_file is not None:
            try:
                stringio = uploaded_file.getvalue().decode("utf-8")
                input_text = stringio
                st.success(f"Fichier '{uploaded_file.name}' chargé avec succès !")
                with st.expander("Voir le contenu du fichier"):
                    st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
            except Exception as e:
                st.error(f"Erreur de lecture du fichier : {e}")

    # Affichage des stats du texte source
    word_count_source = count_words(input_text)
    if input_text:
        st.caption(f"📊 Mots : {word_count_source} | Caractères : {len(input_text)}")
    
    # Bouton d'action centré
    generate_btn = st.button("✨ Générer le Résumé", type="primary", use_container_width=True, disabled=not input_text)

with col2:
    st.subheader("🤖 Résumé Généré")
    
    if generate_btn and input_text:
        with st.spinner("Analyse et synthèse en cours..."):
            try:
                # Appel au modèle
                summary = summarizer.summarize(
                    input_text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    num_beams=num_beams
                )
                
                # Affichage du résultat dans une belle boîte
                st.success("Terminé !")
                st.text_area("Résultat", value=summary, height=300, key="summary_output")
                
                # Statistiques de sortie
                word_count_summary = count_words(summary)
                compression_rate = (1 - (len(summary) / len(input_text))) * 100 if len(input_text) > 0 else 0
                
                # Affichage des métriques
                m1, m2, m3 = st.columns(3)
                m1.metric("Mots", word_count_summary, delta=word_count_summary - word_count_source)
                m2.metric("Compression", f"{compression_rate:.1f}%")
                m3.metric("Temps", "Rapide") # On pourrait mesurer le temps réel avec time.time()
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le résumé",
                    data=summary,
                    file_name="resume_genere.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                logger.error(f"Generation error: {e}")
    
    elif not input_text:
        # État vide (Empty State)
        st.info("👈 Commencez par ajouter du texte ou un fichier dans la colonne de gauche.")
        st.image("https://placehold.co/600x400?text=En+attente+de+contenu", caption="Votre résumé apparaîtra ici.")
