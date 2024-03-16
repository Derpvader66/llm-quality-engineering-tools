import streamlit as st
import pdfplumber
import spacy
from spacy import displacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    all_text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += '\n' + text
    return all_text

def extract_entities(text):
    doc = nlp(text)
    return doc

st.title("Entity Extraction from PDF")

uploaded_file = st.file_uploader("Choose a file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    if st.button('Extract Entities'):
        doc = extract_entities(text)
        html = displacy.render(doc, style="ent")
        st.write(html, unsafe_allow_html=True)
