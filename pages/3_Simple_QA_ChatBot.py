import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

# Streamlit app title
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

History:
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Initialize ConversationChain with the prompt and memory
conversation = ConversationChain(
    llm=chat,
    prompt=prompt,
    memory=ConversationBufferMemory(memory_key="history", input_key="input")
)

# Chat interface
st.header("Chat with the Quality Engineering Assistant")

# Get user input
user_input = st.text_input("You:", "")

if user_input:
    # Pass user input to the conversation chain
    output = conversation.predict(input=user_input)
    
    # Display assistant's response
    st.text_area("Assistant:", value=output, height=200, max_chars=None)