import streamlit as st
from langchain.llms import OpenAI

import langchain.chains
dir(langchain.chains)


# Initialize the OpenAI model with the API key from Streamlit secrets
openai_model = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])

# Set up the ChatCompletionChain with the OpenAI model
#chat_chain = ChatCompletionChain(llm=openai_model)

# Streamlit app layout
st.title('Advanced Chatbot with Langchain and OpenAI')

# Chat history stored in session state to maintain context
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Using a text input to send messages to the chatbot
user_message = st.text_input('You: ', '')

# Displaying the response only when the user sends a message
if user_message:
    # Append the user message to the chat history
    st.session_state['chat_history'].append({"speaker": "user", "text": user_message})
    
    # Generate the response using Langchain and OpenAI
    response = chat_chain.complete(st.session_state['chat_history'])['text']
    
    # Append the bot's response to the chat history
    st.session_state['chat_history'].append({"speaker": "bot", "text": response})
    
    # Display the conversation
    chat_text = "\n".join([f"{msg['speaker'].title()}: {msg['text']}" for msg in st.session_state['chat_history']])
    st.text_area('Chat:', value=chat_text, height=300, disabled=True)
