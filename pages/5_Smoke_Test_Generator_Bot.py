import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

# Streamlit app title
st.title("Manual Test Generator")

# Description
st.write("This bot takes a story, spec or user guide and generates manual test cases from it. To get started, add your document, modify the prompt (if needed) and hit go!")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx", "pdf"])

# Prompt
prompt_text = "Read through the attached Quick Reference Guide word document closely, summarizing the key steps of the business process. Then, write out a detailed manual test case that covers each of the steps, inputs, and expected outputs of the business process. The test case should be written clearly enough that someone unfamiliar with the process could execute it successfully. Make sure to include:\n\n- A descriptive test case name\n- Any prerequisite steps\n- Create realistic test data\n- Step-by-step actions to take\n- Inputs and data to use at each step\n- Expected system responses and outputs at each step\n- Any cleanup steps to reset the state for the next test run\n\nThe test case should cover the normal successful path through the business process.\n\nIn summary, please read through the attached Quick Reference Guide and write a comprehensive manual test case that covers the end-to-end business process flow and key validation points, written clearly enough for someone new to follow and execute."
prompt_input = st.text_area("Prompt", value=prompt_text, height=300)

# OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI chat model
chat = ChatOpenAI(openai_api_key=openai_api_key)

# Create a ConversationChain with the prompt
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_input
)

# Initialize ConversationChain with the prompt and memory
conversation = ConversationChain(
    llm=chat,
    prompt=prompt,
    memory=ConversationBufferMemory(memory_key="history", input_key="input")
)

# Generate test case
if st.button("Submit"):
    if uploaded_file is not None:
        # Read the contents of the uploaded file
        file_contents = uploaded_file.read().decode("utf-8")
        
        # Pass the file contents to the conversation chain
        output = conversation.predict(input=file_contents)
        
        # Display the generated test case
        st.text_area("Generated Test Case", value=output, height=400)
    else:
        st.warning("Please upload a file before submitting.")
