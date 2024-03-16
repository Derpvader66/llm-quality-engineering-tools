import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from PyPDF2 import PdfReader

# Streamlit app title
st.set_page_config(page_title="Quality Engineering Chatbot", page_icon=":memo:")
st.title("Quality Engineering Chatbot")

# OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI chat model
chat = ChatOpenAI(openai_api_key=openai_api_key)

# Create a ConversationChain with a specific prompt
template = """
You are a quality engineering assistant. Your role is to help with various aspects of the quality engineering process.
You can provide guidance on test planning, test case design, defect management, and continuous improvement.
Feel free to ask for more details or clarification if needed.

Business Process Documentation:
{documentation}

History:
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["documentation", "history", "input"],
    template=template
)

# Initialize ConversationChain with the prompt and memory
conversation = ConversationChain(
    llm=chat,
    prompt=prompt,
    memory=ConversationBufferMemory(memory_key="history", input_key="input", human_prefix="Human", ai_prefix="Assistant")
)

# File upload
uploaded_file = st.file_uploader("Upload Business Process Documentation (PDF)", type=["pdf"])

if uploaded_file is not None:
    try:
        # Read the uploaded PDF file
        pdf_reader = PdfReader(uploaded_file)
        documentation_text = ""
        for page in pdf_reader.pages:
            documentation_text += page.extract_text()

        # Display the extracted text
        st.write("Extracted Text:")
        st.write(documentation_text)

        # Chat interface
        st.header("Chat with the Quality Engineering Assistant")

        # Get user input
        user_input = st.text_input("You:", "")

        if user_input:
            # Pass user input and documentation to the conversation chain
            output = conversation.predict(input=user_input, documentation=documentation_text)

            # Display assistant's response
            st.text_area("Assistant:", value=output, height=200, max_chars=None)

        # Clear chat history button
        if st.button("Clear Chat History"):
            conversation.memory.clear()

        # Download test cases button
        if st.button("Download Test Cases"):
            st.download_button(
                label="Download",
                data=output,
                file_name="test_cases.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Error occurred while reading the PDF file: {str(e)}")
else:
    st.write("Please upload a business process documentation file (PDF) to generate test cases.")