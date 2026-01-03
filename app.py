import streamlit as st
from summarizer import summarize_text, load_model
import pdfplumber
import docx

st.set_page_config(page_title="Book Summarizer", layout="wide")

st.title("📚 AI Document Summarizer (BART)")

@st.cache_resource
def get_model():
    return load_model()

tokenizer, model = get_model()

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    return None

st.subheader("🔹 Option 1: Paste Text")
text_input = st.text_area("Enter text", height=200)

st.subheader("🔹 Option 2: Upload Document")
uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"]
)

final_text = ""

if uploaded_file:
    extracted = extract_text(uploaded_file)
    if extracted:
        final_text = extracted
        st.success("File uploaded successfully!")
    else:
        st.error("Unsupported file type")

elif text_input.strip():
    final_text = text_input

if st.button("Summarize"):
    if final_text.strip() == "":
        st.warning("Please upload a file or enter text")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_text(final_text)
            st.subheader("📌 Summary")
            st.success(summary)

            # 🔽 DOWNLOAD BUTTON
            st.download_button(
                label="⬇ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
