import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import chain as chatbot_chain

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

# Inicializa a cadeia do chatbot (se ainda não houver)
if "chain_model" not in st.session_state:
    st.session_state["chain_model"] = chatbot_chain.create_chain()

# Input do usuário
user_input = st.chat_input("Digite sua mensagem aqui...", key="input")
if user_input:
    # 1) Armazena a mensagem humana no estado
    human_msg = HumanMessage(content=user_input)
    st.session_state["messages"].append(human_msg)
    st.session_state["chat_history"].append(human_msg)

    # 2) Obtém as mensagens do banco (lista de SystemMessage) e o raciocínio correspondente
    database_messages, reasoning = chatbot_chain.get_database_responses(
        pergunta_usuario=user_input,
        chat_history=get_historico(),
        nome_colecao=os.getenv("NOME_COLECAO_DB", "qa_excel_collection"),
        diretorio_db=os.getenv("DIRETORIO_DB")
    )

    # 3) Invoca a cadeia LLM, passando o histórico e a lista de SystemMessage
    llm_input = {
        "user_input": user_input,
        "chat_history": get_historico(),
        "database_responses": database_messages  # AGORA é lista[SystemMessage]
    }
    llm_response = st.session_state["chain_model"].invoke(llm_input)

    # 4) Armazena a resposta do LLM no estado
    ai_msg = AIMessage(content=llm_response)
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
