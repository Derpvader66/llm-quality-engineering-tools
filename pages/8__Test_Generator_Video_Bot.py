import cv2
import streamlit as st
import os

# Function to extract frames from a video
def extract_frames(video_file):
    # Open the video file
    video_capture = cv2.VideoCapture(video_file)
    
    # Initialize frame counter
    frame_count = 0
    
    # Loop through each frame in the video
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        
        # Check if frame was read successfully
        if not ret:
            break
        
        # Process the frame (e.g., display it in the Streamlit app)
        st.image(frame, channels="BGR")
        
        # Increment frame counter
        frame_count += 1

# Streamlit app title
st.title("Manual Test Generator")

# Description
st.write("This bot takes a story, spec or user guide and generates manual test cases from it. To get started, add your document, modify the prompt (if needed) and hit go!")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx", "pdf", "mp4", "avi", "mkv"])

# Prompt
prompt_text = "Read through the attached Quick Reference Guide word document closely, summarizing the key steps of the business process. Then, write out a detailed manual test case that covers each of the steps, inputs, and expected outputs of the business process. The test case should be written clearly enough that someone unfamiliar with the process could execute it successfully. Make sure to include:\n\n- A descriptive test case name\n- Any prerequisite steps\n- Create realistic test data\n- Step-by-step actions to take\n- Inputs and data to use at each step\n- Expected system responses and outputs at each step\n- Any cleanup steps to reset the state for the next test run\n\nThe test case should cover the normal successful path through the business process.\n\nIn summary, please read through the attached Quick Reference Guide and write a comprehensive manual test case that covers the end-to-end business process flow and key validation points, written clearly enough for someone new to follow and execute."
prompt_input = st.text_area("Prompt", value=prompt_text, height=300)

# OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI chat model
chat = ChatOpenAI(openai_api_key=openai_api_key)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["input"],
    template=prompt_input
)

# Initialize LLMChain with the prompt
chain = LLMChain(llm=chat, prompt=prompt)

# Generate test case
if st.button("Submit"):
    if uploaded_file is not None:
        # Read the contents of the uploaded file
        file_contents = uploaded_file.read().decode("utf-8")
        
        # Check if the file is a video
        if uploaded_file.type.startswith('video/'):
            # Display video player
            st.video(uploaded_file)
            
            # Extract frames
            extract_frames(uploaded_file)
        else:
            # Pass the file contents to the LLMChain
            output = chain.run(input=file_contents)
            
            # Display the generated test case
            st.text_area("Generated Test Case", value=output, height=400)
    else:
        st.warning("Please upload a file before submitting.")
