import streamlit as st
from langchain_community import chat_models

import chatbot


@st.cache_resource
def get_chatbot() -> chatbot.Chatbot:
    if 'chatbot' not in st.session_state:
        chat_model = chat_models.ChatOllama(model='llama3:instruct')
        st.session_state.chatbot = chatbot.Chatbot.from_chat_model(chat_model)
    return st.session_state.chatbot


chatbot = get_chatbot()

st.sidebar.button('Clear History', on_click=chatbot.clear_messages)

for message in chatbot.messages:
    with st.chat_message(name=message.type):
        st.write(message.content)

if question := st.chat_input():
    with st.chat_message(name='human'):
        st.write(question)
    with st.chat_message(name='ai'):
        with st.spinner('Responding...'):
            st.write_stream(chatbot.stream(question))
