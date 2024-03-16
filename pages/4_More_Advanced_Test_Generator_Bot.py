import streamlit as st
from openai import ChatCompletion

st.set_page_config(page_title="Test Case Generator", page_icon=":memo:")
st.title("Test Case Generator")

openai_api_key = st.secrets["OPENAI_API_KEY"]
model_name = "gpt-4"

uploaded_file = st.file_uploader("Upload Business Process Document (PDF)", type=["pdf"])

test_cases = ""  # Initialize the test_cases variable

if uploaded_file is not None:
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(uploaded_file)
        document_text = ""
        for page in pdf_reader.pages:
            document_text += page.extract_text()
        st.write("Uploaded Document:")
        st.write(document_text)
    except Exception as e:
        st.error(f"Error occurred while reading the PDF file: {str(e)}")

    prompt = f"Generate test cases for the following business process document:\n\n{document_text}\n\nTest Cases:"
    
    if st.button("Generate Test Cases"):
        try:
            response = ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates test cases."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.7,
            )
            test_cases = response.choices[0].message['content'].strip()  # Update the test_cases variable
            st.write("Generated Test Cases:")
            st.write(test_cases)
        except Exception as e:
            st.error(f"Error occurred while generating test cases: {str(e)}")

    if test_cases:
        st.download_button(
            label="Download Test Cases",
            data=test_cases,
            file_name="test_cases.txt",
            mime="text/plain"
        )