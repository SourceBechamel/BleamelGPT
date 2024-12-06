import streamlit as st

from bleamel.chatbots import factory, protocol


@st.cache_resource
def get_chatbot() -> protocol.ChatbotProtocol:
    if 'chatbot' not in st.session_state:
        bot = factory.FlowerChatbotFactory.create_basic_chatbot()
        st.session_state.chatbot = bot
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
