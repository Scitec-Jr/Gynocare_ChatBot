from langchain_core.prompts import MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

def create_chain(temperature: float = 0.1) -> RunnableSerializable:
    llm: ChatGroq = ChatGroq(
        model="qwen-qwq-32b",
        temperature=temperature
    )

    system_prompt: str = """
1. Você é um atendente da Clinica Gynocare. 
2. Suas respostas devem ser simples e diretas. Com linguagem formal, mas gentil.
3. Você receberá algumas respostas similares que foram encontradas no nosso banco de dados de perguntas frequentes. Você deve usar essas respostas como base para suas respostas, caso não encontre uma resposta adequada, você deve responder com "Desculpe, não tenho uma resposta para isso no momento. Por favor, entre em contato com nosso suporte ao cliente para mais informações.".
4. Você deve responder apenas com o texto, sem formatação, sem emojis e sem links.
"""

    user_prompt: str = "{user_input}"

    prompt_template:ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="database_responses"),
        ('user', user_prompt)
    ])

    chain: RunnableSerializable = prompt_template | llm | StrOutputParser()
    return chain

if 'chain' not in st.session_state:
    st.session_state['chain'] = create_chain()

def get_database_responses(user_input: str):
    pass

def talk_to_chain(user_input: str):
    return st.session_state['chain'].invoke({
        'user_input': user_input,
        'chat_history': st.session_state['chat_history'],
        'database_responses': get_database_responses(user_input)
    })

