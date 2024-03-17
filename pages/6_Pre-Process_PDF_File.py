import streamlit as st
import pdfplumber
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initializing the stopwords set outside the function for efficiency
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and non-alphanumeric characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token.isalnum()]
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to extract text from PDF
@st.cache(allow_output_mutation=True)
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"An error occurred while extracting text: {e}")
    return text

def display_preprocessed_text(uploaded_file):
    # Extract text from uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        # Preprocess the extracted text
        preprocessed_text = preprocess_text(pdf_text)
        
        # Display preprocessed text
        st.subheader("Preprocessed Text:")
        st.write(preprocessed_text)
    else:
        st.warning("No text could be extracted from the uploaded file.")

def main():
    # Streamlit app title
    st.title("PDF Text Preprocessing")
    
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        display_preprocessed_text(uploaded_file)

if __name__ == "__main__":
    main()
