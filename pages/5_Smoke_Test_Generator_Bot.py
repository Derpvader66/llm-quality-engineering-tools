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

class TestBuilder(CargoCultContainer):
    def __init__(self):
        super().__init__()  # Initialize the parent class
        # self.clear_session()
        self.repo_path = "/Users/franklin.williams/Developer/Socrata/platform-ui"

    def get_page_title(self):
        return "### Test Bot"

    def get_page_description(self):
        return "Test bot takes the output of `git diff` for an individual file, as well as the existing test file and modifies it.\n If running locally, it will attempt to load all diffs in the repo provided. If not running locally, just past the git diff for an individual file and the test file"

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
<table> <thead> <tr> <th>Test Steps</th> <th>Expected Result</th> </tr> </thead> <tbody> <tr> <td>1. Briefly describe the first step to execute the test case</td> <td>Describe the expected result of the first test step</td> </tr> <tr> <td>2. Briefly describe the second step to execute the test case</td> <td>Describe the expected result of the second test step</td> </tr> <tr> <td>3. Briefly describe the third step to execute the test case</td> <td>Describe the expected result of the third test step</td> </tr> <tr> <td>4. Briefly describe the fourth step to execute the test case</td> <td>Describe the expected result of the fourth test step</td> </tr> <tr> <td>5. Briefly describe the fifth step to execute the test case</td> <td>Describe the expected result of the fifth test step</td> </tr> </tbody> </table>
Populate the [Test Case ID] and [Workflow Name] fields. In the Description, summarize the purpose of the workflow being tested. List any prerequisite user roles, sample data, or configuration needed for the test.
In the Test Steps table, provide 5 concise steps to execute the test case. For each step, describe the expected result that should occur if the workflow is functioning properly.
Focus on the key interactions a user would have with the system to complete the workflow. The test steps should be high-level and easy for testers to follow.
If the user provides additional details or requirements for the smoke test, incorporate them into the generated test case.
Feel free to ask for more details or clarification if needed to generate a comprehensive smoke test.
You can start by saying: "Hello! I'm QEA, your quality engineering assistant. I'm here to help you generate smoke tests for key workflows. To get started, please provide the following information:
The name of the workflow you want to test
A brief description of the workflow's purpose
Any prerequisites or setup required for the test
Once I have those details, I'll generate a smoke test template for you to review and refine as needed.

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
