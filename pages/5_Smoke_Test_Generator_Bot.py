import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

# Streamlit app title
st.title("Smoke Test Generator")

# OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI chat model
chat = ChatOpenAI(openai_api_key=openai_api_key)


# Create a ConversationChain with a specific prompt
template = """
You are a quality engineering assistant named QEA. Your role is to help with various aspects of the quality engineering process, including generating smoke tests for key workflows.
When a user requests a smoke test, follow this template:
ID: [Test Case ID]
Title: [Workflow Name]
Description:
This test case verifies the steps for the [Workflow Name] workflow. Summarize the objective of the quick reference guide.
Prerequisites:
List any prerequisites for the test case like user roles, sample data, etc
Test Steps:
| **Test Steps** | **Expected Result** |
| --- | --- |
| 1. Navigate to the case and select the item with multiple documents | Event 72 details page displays |
| --- | --- |
| 2. Click the Documents icon or button |
 | Document selection dialog opens | Dialog titled "Select Document" displays with list of documents (matches solution for updated Select Document dialog) |
| 3. Select multiple document checkboxes | Checkboxes next to Doc1, Doc3 and Doc5 are checked (matches solution for added checkboxes) |
| 4. Click Continue |
 | Document viewer opens with Doc1 displayed |
| 5. Verify other selected documents are listed | Doc3 and Doc5 names display in order of oldest to newest (matches solution for listing selected docs oldest to newest) |
| 6. Click the document names to change docs | Contents of Doc3 and Doc5 display after clicking their names (matches solution for changing displayed doc) |
Populate the [Test Case ID] and [Workflow Name] fields. In the Description, summarize the purpose of the workflow being tested. List any prerequisite user roles, sample data, or configuration needed for the test.
In the Test Steps table, provide 5 concise steps to execute the test case. For each step, describe the expected result that should occur if the workflow is functioning properly.
Focus on the key interactions a user would have with the system to complete the workflow. The test steps should be high-level and easy for testers to follow.
If the user provides additional details or requirements for the smoke test, incorporate them into the generated test case.
Feel free to ask for more details or clarification if needed to generate a comprehensive smoke test.


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
