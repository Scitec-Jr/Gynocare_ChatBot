from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import banco  # Nosso módulo banco.py
model = "llama-3.1-8b-instant"

def reformular_pergunta(
    chat_history: list[HumanMessage | AIMessage],
    pergunta_atual: str,
    temperature: float = 0.0
) -> str:
    """
    Usa o modelo LLM para reformular a pergunta do usuário levando em consideração o histórico.

    Args:
        chat_history: Lista de mensagens anteriores (HumanMessage e AIMessage).
        pergunta_atual: Pergunta atual do usuário.
        temperature: Temperatura para o modelo de reformulação.

    Returns:
        Pergunta reformulada como string.
    """

    llm_rewriter = ChatGroq(model=model, temperature=temperature)
    rewriter_prompt = (
        "Você é um assistente que pega uma pergunta de usuário e seu histórico de conversa e "
        "reformula para ser uma consulta independente. Considere apenas fatos, sem suposições. "
        "Retorne apenas a pergunta reformulada.\n\n"
        "Histórico de conversa:\n{history}\n"
        "Pergunta do usuário: {pergunta}\n"
        "Pergunta reformulada:"
    )

    formatted_history: list[str] = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"Usuário: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_history.append(f"Assistente: {msg.content}")
    history_text = "\n".join(formatted_history)

    prompt_input = rewriter_prompt.format(history=history_text, pergunta=pergunta_atual)
    response = llm_rewriter.predict(prompt_input)  # assume que predict retorna string
    return response.strip()


def get_database_responses(
    pergunta_usuario: str,
    chat_history: list[HumanMessage | AIMessage],
    nome_colecao: str = "qa_excel_collection",
    diretorio_db: str | None = None,
    k: int = 3
) -> tuple[list[SystemMessage], dict]:
    """
    Busca as k entradas mais similares no banco de dados e formata para a LLM.

    Args:
        pergunta_usuario: Pergunta do usuário.
        chat_history: Histórico de conversa para reformulação.
        nome_colecao: Nome da coleção no ChromaDB.
        diretorio_db: Diretório de persistência do ChromaDB.
        k: Número de resultados a retornar.

    Returns:
        Tuple contendo:
          - database_messages: lista de SystemMessage contendo as tabelas Markdown encontradas
          - reasoning: Dicionário com detalhes de depuração (perguntas originais, reformuladas e distâncias).
    """
    # 1. Reformular a pergunta para consulta independente
    pergunta_reformulada = reformular_pergunta(chat_history, pergunta_usuario)

    # 2. Consultar o banco usando a pergunta reformulada
    resultados = banco.buscar_perguntas_no_banco(
        pergunta_usuario=pergunta_reformulada,
        nome_colecao=nome_colecao,
        diretorio_db=diretorio_db,
        n_results=k
    )

    # 3. Montar string de respostas (tabelas) e registrar raciocínio
    partes_markdown: list[str] = []
    reasoning: dict = {
        "pergunta_original": pergunta_usuario,
        "pergunta_reformulada": pergunta_reformulada,
        "matches": []  # lista de {rank, pergunta_base, distancia, tabela}
    }

    if resultados:
        for idx, (pergunta_base, tabela, distancia) in enumerate(resultados):
            partes_markdown.append(
                f"### Resultado {idx+1}: {pergunta_base} (Distância: {distancia:.4f})\n{tabela}"
            )
            reasoning["matches"].append({
                "rank": idx + 1,
                "pergunta_base": pergunta_base,
                "distancia": distancia,
                "tabela": tabela
            })
        texto_completo = "\n\n".join(partes_markdown)
    else:
        texto_completo = "Nenhum resultado encontrado."

    # Envolvemos o texto em um único SystemMessage (poderíamos separar por rank, mas um único bloco é suficiente)
    database_messages = [SystemMessage(content=texto_completo)]

    return database_messages, reasoning


def create_chain(temperature: float = 0.1) -> ChatPromptTemplate:
    """
    Cria e retorna a cadeia (chain) de chamadas para o ChatGroq.

    Args:
        temperature: Temperatura para o LLM.

    Returns:
        Um objeto ChatPromptTemplate para ser invocado posteriormente.
    """
    system_prompt = (
        "Você é um atendente da Clínica Gynocare. Suas respostas devem ser simples e diretas, "
        "com linguagem formal e gentil. Você receberá respostas similares do banco de perguntas "
        "frequentes (tabelas por idade). Use essas respostas como base. Se não encontrar, responda:\n"
        "'Desculpe, não tenho uma resposta para isso no momento. Por favor, entre em contato com "
        "nosso suporte ao cliente para mais informações.' Responda apenas com texto, sem formatação, "
        "emojis ou links."
    )

    user_prompt = (
        "Dados do banco (tabelas de idade/resposta):\n{database_responses}\n\n"
        "Pergunta do usuário: {user_input}\n\n"
        "Com base no histórico e nesses dados, responda de forma adequada."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="database_responses"),
        ("user", user_prompt)
    ])

    llm = ChatGroq(model=model, temperature=temperature)
    chain = prompt_template | llm | StrOutputParser()
    return chain
