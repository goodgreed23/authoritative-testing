import streamlit as st
from openai import OpenAI
import pandas as pd
import os
import utils.prompt_utils as prompt_utils
import openpyxl

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from gcloud import storage
# from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
# from google.oauth2.service_account import Credentials

from models import MODEL_CONFIGS
from utils.prompt_utils import target_styles, definitions, survey_items
from utils.eval_qs import TA_0s, TA_100s
from utils.utils import response_generator
from datetime import datetime

import shutil

st.set_page_config(page_title="Therapist Chatbot Evaluation", page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)

# CONFIGS
style_id = 0
min_turns = 20   # number of turns to make before users can save the chat
MODEL_SELECTED = "gpt-4o"

# Show title and description.
st.title("Therapist Chatbot Evaluation")

# Get participant ID 
user_PID = st.text_input("What is your participant ID?")

# Retrieve API key from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize GCP credentials and bucket details
credentials_dict = {
    'type': st.secrets.gcs["type"],
    'client_id': st.secrets.gcs["client_id"],
    'client_email': st.secrets.gcs["client_email"],
    'private_key': st.secrets.gcs["private_key"],
    'private_key_id': st.secrets.gcs["private_key_id"],
}
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict)
client = storage.Client(credentials=credentials, project='digital-sprite-450023-c5')
bucket = client.get_bucket('coco-streamlit-bucket')
file_name = 'NA'

def save_duration():
    if st.session_state["start_time"]:
        duration = datetime.now() - st.session_state["start_time"]
        st.session_state["evaluation_durations"] = duration
    return duration

if not user_PID:
    st.info("Please enter your participant ID to start.", icon="🗝️")
else:
    st.write("""**Start chatting with the AI therapist. After getting >= 10 responses from the therapist, a 'save' button will appear. After you finish the conversation naturally,
         you may click the 'save' button to save the conversation and then fill out the evaluation questions.**""")
    
    # Add a slider for adaptation intensity (0 = no adaptation, 1 = full adaptation)
    adaptation_intensity = st.slider("Adaptation Intensity", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                     help="Adjust the intensity of the style adaptation. Lower values preserve more of the original response.")
    preservation_percentage = int(100 * (1 - adaptation_intensity))
    
    # Create an OpenAI client.
    llm = ChatOpenAI(model=MODEL_SELECTED, api_key=openai_api_key)
    
    # Start tracking the duration.
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = datetime.now()
    if "evaluation_durations" not in st.session_state:
        st.session_state["evaluation_durations"] = None
    start_time_row = pd.DataFrame([{"role": "Start Time", "content": st.session_state['start_time']}])
        
    # Therapist agent
    therapist_model_config = MODEL_CONFIGS['Therapist']
    therapyagent_prompt_template = ChatPromptTemplate.from_messages([
        ("system", therapist_model_config['prompt']),
        MessagesPlaceholder(variable_name="history"),  # dynamic insertion of past conversation history
        ("human", "{input}"),
    ])
    
    # Communication style modifier prompt
    modifier_model_config = MODEL_CONFIGS['Modifier']
    # Update the prompt template to include the new variables:
    # "communication_style", "definition", "survey_item", "adaptation_intensity",
    # "preservation_percentage", "unadapted_chat_history", "unadapted_response"
    csm_prompt_template = PromptTemplate(
        variables=["communication_style", "definition", "survey_item", "adaptation_intensity", "preservation_percentage", "unadapted_chat_history", "unadapted_response"],
        template=modifier_model_config['prompt']
    )

    # Set up Streamlit history memory.
    msgs = StreamlitChatMessageHistory(key="chat_history")

    # Create a session state variable to store the chat messages (persisting across reruns).
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """Hello, I am an AI therapist, here to support you in navigating the challenges and emotions you may face as a caregiver. 
             Is there a specific caregiving challenge or experience you would like to share with me today?"""}
        ]

    # Display the existing chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat_history_df = pd.DataFrame(st.session_state.messages)
    
    # Create a chat input field for the user.
    if user_input := st.chat_input("Enter your input here."):
        # Create a therapy chatbot LLM chain.
        therapyagent_chain = therapyagent_prompt_template | llm
        therapy_chain_with_history = RunnableWithMessageHistory(
            therapyagent_chain,
            lambda session_id: msgs,  # Always return the instance created earlier
            input_messages_key="input",
            history_messages_key="history",
        )

        # Create a CSM chain.
        csmagent_chain = LLMChain(
            llm=llm,
            prompt=csm_prompt_template,
            verbose=False,
            output_parser=StrOutputParser()
        )

        # Append user input to the session messages and display it.
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"session_id": "any"}}
        unada_response = therapy_chain_with_history.invoke({"input": user_input}, config)
        unada_bot_response = unada_response.content

        target_style = target_styles[style_id]
        definition = definitions[style_id]
        survey_item = survey_items[style_id]
        
        # Generate the adapted response using the new parameters.
        ada_response = csmagent_chain.predict(
            communication_style=target_style,
            definition=definition,
            survey_item=survey_item,
            adaptation_intensity=adaptation_intensity,
            preservation_percentage=preservation_percentage,
            unadapted_chat_history=st.session_state.messages,
            unadapted_response=unada_bot_response
        )

        # Display the adapted response.
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(response=ada_response))
        
        st.session_state.messages.append({"role": "assistant", "content": ada_response})
        chat_history_df = pd.DataFrame(st.session_state.messages)

    # Automatically save the conversation after reaching the minimum number of turns or if the user types "save"/"stop".
    if chat_history_df.shape[0] >= min_turns or (user_input and user_input.lower() in ["save", "stop"]):
        file_name = "{style}_P{PID}.csv".format(style=target_styles[style_id], PID=user_PID)
        created_files_path = "conv_history_P{PID}".format(PID=user_PID)
        if not os.path.exists(created_files_path):
            os.makedirs(created_files_path)
                
        end_time_row = pd.DataFrame([{"role": "End Time", "content": datetime.now()}])
        duration_row = pd.DataFrame([{"role": "Duration", "content": save_duration()}])
        chat_history_df = pd.concat([chat_history_df, start_time_row, end_time_row, duration_row], ignore_index=True)
        
        chat_history_df.to_csv(os.path.join(created_files_path, file_name), index=False)
        
        blob = bucket.blob(file_name)
        blob.upload_from_filename(os.path.join(created_files_path, file_name))
        shutil.rmtree(created_files_path)
        
        if st.button("Save Conversation & Start Evaluation"):
            st.write("**Chat history is saved successfully. You can begin filling out the evaluation questions now.**")
            st.cache_data.clear()
