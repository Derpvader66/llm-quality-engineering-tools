import streamlit as st
import os
import shutil

# Define the directory where the uploaded files will be saved
TEMPLATES_DIR = "templates/test_case"

def save_test_case_template_files(test_case_template_files):
    if test_case_template_files:
        # Remove the existing directory and its contents
        if os.path.exists(TEMPLATES_DIR):
            shutil.rmtree(TEMPLATES_DIR)
        
        # Create a new directory for the uploaded files
        os.makedirs(TEMPLATES_DIR)
        
        # Save each uploaded file to the directory
        for uploaded_file in test_case_template_files:
            file_path = os.path.join(TEMPLATES_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded {len(test_case_template_files)} file(s) successfully!")
    else:
        st.warning("No files uploaded.")

def main():
    st.title("Test Case Generator")
    
    # Template upload section
    st.subheader("Test Case Template")
    
    if os.path.exists(TEMPLATES_DIR):
        # Display the list of uploaded files
        uploaded_files = os.listdir(TEMPLATES_DIR)
        if uploaded_files:
            st.write("Uploaded files:")
            for file_name in uploaded_files:
                st.write(f"- {file_name}")
        else:
            st.write("No files uploaded yet.")
    
    # File uploader for test case templates
    template_files = st.file_uploader(
        "Upload your test case templates (Word documents):",
        type=["docx"],
        accept_multiple_files=True
    )
    
    # Save the uploaded files when the user clicks the "Upload" button
    if st.button("Upload"):
        save_test_case_template_files(template_files)

if __name__ == "__main__":
    main()