import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Function to extract text content from PDF files
def extract_text_from_pdfs(uploaded_files):
    texts = []
    for pdf_file in uploaded_files:
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or " "
        texts.append(text.strip())
    return texts

# Input for specifying labels
label_list = st.text_input("Enter labels separated by commas").split(',')
label_list = [label.strip() for label in label_list if label.strip()]  # Clean and filter empty labels

if label_list and len(set(label_list)) > 1:
    # Define file upload component
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        texts = extract_text_from_pdfs(uploaded_files)

        # Create a select box for each uploaded file to assign labels
        labels = [st.selectbox(f"Label for {pdf_file.name}:", label_list) for pdf_file in uploaded_files if texts[uploaded_files.index(pdf_file)].strip()]

        # Vectorize text using TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform(texts)

        # Map labels to numeric values
        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        y = [label_mapping[label] for label in labels]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM classifier
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = svm_classifier.predict(X_test)
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    else:
        st.info("Upload PDF files to start the classification process.")
else:
      st.error("Please specify at least two unique labels.")
