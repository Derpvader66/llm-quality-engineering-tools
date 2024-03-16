import streamlit as st
import os
import shutil

def save_test_case_template_file(test_case_template_file):
   #
   st.Write(test_case_template_file.name)

def main():
   st.title("Test Case Generator")
   # Template upload section
   st.subheader("Test Case Template")
   template_file = st.file_uploader("Upload your test case template (Word document):", type=["docx"])
   save_test_case_template_file(template_file)

if __name__ == "__main__":
   main()