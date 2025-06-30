import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

import requests

load_dotenv()

# Configuração da página
st.set_page_config(page_title="ChatBot Gynocare", page_icon=":robot_face:", layout="wide")
st.markdown("# ChatBot Gynocare 🤖")

# --- Estado da Sessão ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # armazena instâncias de HumanMessage | AIMessage
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # lista de mensagens para chain
if "reasonings" not in st.session_state:
    st.session_state["reasonings"] = []  # lista de dicts com raciocínio de cada rodada

# Recupera o histórico de chat (lista de BaseMessage)
def get_historico() -> list[HumanMessage | AIMessage]:
    return st.session_state["chat_history"]

# Exibe as mensagens na interface
def exibir_messages() -> None:
    for message in st.session_state["messages"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="assistant"):
                st.markdown(message.content)

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

# Input do usuário
user_input = st.chat_input("Digite sua mensagem aqui...", key="input")
if user_input:
    # 1) Armazena a mensagem humana no estado
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    st.session_state["chat_history"].append(human_msg)

    # 2) Chama a API
    hist_payload = []
    for msg in get_historico():
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        hist_payload.append({"role": role, "content": msg.content})
    payload = {"user_input": user_input, "chat_history": hist_payload}
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        resposta_texto = data.get("response", "")
        reasoning = data.get("reasoning", {})
    except Exception as e:
        resposta_texto = "Erro ao consultar a API."
        reasoning = {}

    # 3) Armazena a resposta no estado
    ai_msg = AIMessage(content=resposta_texto)
    st.session_state["messages"].append(ai_msg)
    st.session_state["chat_history"].append(ai_msg)
    st.session_state["reasonings"].append(reasoning)

    # Recarrega a página para exibir tudo
    st.rerun()

# Exibe histórico de mensagens
exibir_messages()

# Exibe raciocínio no expander para a última interação
if st.session_state["reasonings"]:
    ultima_raz = st.session_state["reasonings"][-1]
    with st.expander("Ver raciocínio para esta resposta"):
        st.markdown(f"**Pergunta original:** {ultima_raz['pergunta_original']}")
        st.markdown(f"**Pergunta reformulada:** {ultima_raz['pergunta_reformulada']}")
        st.markdown("**Resultados obtidos do banco:**")
        for match in ultima_raz["matches"]:
            st.markdown(
                f"- Rank {match['rank']}: {match['pergunta_base']} (Distância: {match['distancia']:.4f})"
            )
            st.markdown(match["tabela"])
