import streamlit as st
from summarizer import summarize_text, load_model

st.set_page_config(page_title="Book Summarizer", layout="wide")

st.title("📚 Book Summarizer using BART")

@st.cache_resource
def get_model():
    return load_model()

# Load model once
tokenizer, model = get_model()

text = st.text_area(
    "Paste Book Text Here",
    height=300,
    placeholder="Paste your book chapter or long text here..."
)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
            st.subheader("Summary")
            st.success(summary)
