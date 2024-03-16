import streamlit as st
import os
import shutil
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
# Set page title
st.set_page_config(page_title="Test Case Generator", page_icon=":memo:")
# App title and description
st.title("Test Case Generator")
st.markdown("Upload your business process documents and user documentation files to generate test cases.")
openai_api_key = st.secrets["OPENAI_API_KEY"]

example_test_case = """
Test Case ID: TC001
Description: Verify user login functionality
Prerequisites:

Do some Stuff before you do other stuff

Steps:
 Step 1
"""
#st.header("Example Test Case Format")
#st.write(example_test_case)

uploaded_business_files = st.file_uploader("Choose business process documents", accept_multiple_files=True)
uploaded_user_files = st.file_uploader("Choose user documentation files", accept_multiple_files=True)

def process_files(uploaded_files, directory):
   if uploaded_files:
       if os.path.exists(directory):
           shutil.rmtree(directory)
       os.makedirs(directory)
       for uploaded_file in uploaded_files:
           with open(os.path.join(directory, uploaded_file.name), "wb") as f:
               f.write(uploaded_file.getbuffer())
       st.success(f"{len(uploaded_files)} file(s) uploaded successfully to {directory}!")
# Generate test cases
       
def generate_test_cases():
   # Load and process the documents
   business_docs = UnstructuredFileLoader(["business_process_docs/" + f for f in os.listdir("business_process_docs")]).load()
   user_docs = UnstructuredFileLoader(["user_documentation/" + f for f in os.listdir("user_documentation")]).load()
   docs = business_docs + user_docs
   text_splitter = CharacterTextSplitter(chunk_size=4096, chunk_overlap=0)
   texts = text_splitter.split_documents(docs)
   embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
   db = FAISS.from_documents(texts, embeddings)
   chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key, temperature=0), chain_type="stuff")
   query = f"Summarize the business process.  List the activities that make up the process.  Do not include any steps for an activity.\n"
   bp_summary = chain.run(input_documents=db.similarity_search(query), question=query)

   query = f"Generate a detailed test case based on business process and user documentation. DO NOT INCLUDE THE EXAMPLE FORMAT IN THE OUTPUT."
   test_cases = chain.run(input_documents=db.similarity_search(query), question=query)
   st.header("Generated Test Cases")
   st.write(test_cases)
   shutil.rmtree("business_process_docs")
   shutil.rmtree("user_documentation")

process_files(uploaded_business_files, "business_process_docs")
process_files(uploaded_user_files, "user_documentation")

if st.button("Generate Test Cases"):
   if openai_api_key:
       generate_test_cases()
   else:
       st.warning("Please set OpenAI API Key in the StreamLit Secrets Manager")