import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

st.set_page_config(page_title="ChatBot Gynocare", page_icon=":robot_face:", layout="wide")
st.markdown("# ChatBot Gynocare :robot_face:")

if "messages" not in st.session_state:
    st.session_state['messages'] = []

def print_messages():
    for message in st.session_state['messages']:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="assistant"):
                st.markdown(message.content)                

st.chat_input("Digite sua mensagem aqui...", key="input")
if st.session_state['input']:
    user_message = HumanMessage(content=st.session_state['input'])
    bot_response = AIMessage(content="Esta é uma resposta simulada do ChatBot.")
    st.session_state['messages'].append(user_message)
    st.session_state['messages'].append(bot_response)
    st.rerun()
