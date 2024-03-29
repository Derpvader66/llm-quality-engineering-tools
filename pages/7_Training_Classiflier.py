import streamlit as st
import PyPDF2
import io
import spacy

# Load the English tokenizer, tagger, parser, NER, and word vectors
# Ensuring we're using the latest English model available for spaCy
nlp = spacy.load("en_core_web_trf")

def extract_text_from_pdf(document):
    """
    Extract text from each page of the uploaded PDF document using PyPDF2.
    
    :param document: The uploaded PDF document.
    :return: Extracted text.
    """
    text = ""
    # Read the PDF file
    pdfReader = PyPDF2.PdfReader(io.BytesIO(document.read()))
    # Iterate through each page and extract text
    for page_num in range(len(pdfReader.pages)):
        page = pdfReader.get_page(page_num)  # Using get_page for newer versions
        text += page.extract_text()
    return text

def identify_actions(text):
    """
    Identify potential actions in the text using NLP.
    
    :param text: The text to process.
    :return: A list of actions and targets.
    """
    actions = []
    doc = nlp(text)
    for sent in doc.sents:
        action = None
        target = None

        for token in sent:
            if token.pos_ == "VERB":
                action = token.text
                for child in token.children:
                    if child.dep_ == "dobj":
                        target = child.text
                        break

        if action and target:
            actions.append(f"Action: {action}, Target: {target}")
    return actions

# Streamlit app
st.title("Business Process Step Extractor")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Display extracted text (optional)
    st.text_area("Extracted Text", extracted_text, height=250)

    # Identify actions in the extracted text
    actions = identify_actions(extracted_text)

    if actions:
        st.subheader("Identified Actions")
        for action in actions:
            st.write(action)
    else:
        st.write("No actions identified.")
