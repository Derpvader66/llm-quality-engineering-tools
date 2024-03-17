import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Function to extract text content from PDF files
def extract_text_from_pdf(pdf_file):
    text = ""
    for page in pdf_file.pages:
        text += page.extract_text()
    return text

# Define file upload component
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    # Extract text content from uploaded PDF files
    texts = []
    labels = []
    for pdf_file in uploaded_files:
        pdf_reader = PdfReader(pdf_file)
        text = extract_text_from_pdf(pdf_reader)
        texts.append(text)
        labels.append(st.text_input(f"Label for {pdf_file.name}:"))

    # Convert labels to numeric values
    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_mapping[label] for label in labels]

    # Vectorize text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(texts)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Train SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = svm_classifier.predict(X_test)
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
