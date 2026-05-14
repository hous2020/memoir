import logging

import streamlit as st

from summarizer import FrenchSummarizer


st.set_page_config(
    page_title="Resume Automatique - Modele Scratch",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_words(text):
    if not text:
        return 0
    return len(text.split())


@st.cache_resource
def load_summarizer():
    return FrenchSummarizer(model_type="scratch", model_path="models/transformer_scratch.pth")


with st.sidebar:
    st.header("Parametres")
    st.divider()
    st.subheader("Modele")
    st.info("Cette application utilise uniquement votre modele Transformer entraine from scratch.")

    max_length = st.slider("Longueur maximale", 32, 256, 128)
    min_length = 10
    num_beams = 1

    st.markdown("---")
    st.caption("Modele from scratch")


st.title("Resume Automatique de Documents")
st.markdown("Generez des resumes avec votre propre modele entraine depuis zero.")

try:
    with st.spinner("Initialisation du modele scratch..."):
        summarizer = load_summarizer()
except Exception as exc:
    st.error(f"Erreur critique lors du chargement du modele : {exc}")
    st.stop()


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Document source")

    tab_text, tab_file = st.tabs(["Saisie manuelle", "Importer un fichier"])
    input_text = ""

    with tab_text:
        text_area_val = st.text_area(
            "Collez votre texte ici",
            height=400,
            placeholder="Copiez-collez ici le contenu d'un article, d'un rapport...",
            key="input_text_area",
        )
        if text_area_val:
            input_text = text_area_val

    with tab_file:
        uploaded_file = st.file_uploader("Choisissez un fichier texte (.txt, .md)", type=["txt", "md"])
        if uploaded_file is not None:
            try:
                input_text = uploaded_file.getvalue().decode("utf-8")
                st.success(f"Fichier '{uploaded_file.name}' charge avec succes.")
                with st.expander("Voir le contenu du fichier"):
                    preview = input_text[:1000] + "..." if len(input_text) > 1000 else input_text
                    st.text(preview)
            except Exception as exc:
                st.error(f"Erreur de lecture du fichier : {exc}")

    word_count_source = count_words(input_text)
    if input_text:
        st.caption(f"Mots : {word_count_source} | Caracteres : {len(input_text)}")

    generate_btn = st.button(
        "Generer le resume",
        type="primary",
        use_container_width=True,
        disabled=not input_text,
    )

with col2:
    st.subheader("Resume genere")

    if generate_btn and input_text:
        with st.spinner("Generation en cours..."):
            try:
                summary = summarizer.summarize(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                )

                st.success("Termine.")
                st.text_area("Resultat", value=summary, height=300, key="summary_output")

                word_count_summary = count_words(summary)
                compression_rate = (1 - (len(summary) / len(input_text))) * 100 if input_text else 0

                m1, m2 = st.columns(2)
                m1.metric("Mots", word_count_summary, delta=word_count_summary - word_count_source)
                m2.metric("Compression", f"{compression_rate:.1f}%")

                st.download_button(
                    label="Telecharger le resume",
                    data=summary,
                    file_name="resume_genere.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Une erreur est survenue : {exc}")
                logger.error(f"Generation error: {exc}")
    elif not input_text:
        st.info("Commencez par ajouter du texte ou un fichier dans la colonne de gauche.")
