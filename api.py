from typing import List, Dict
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import chain as chatbot_chain

app = FastAPI()

# Carrega chain na inicialização
chat_model = chatbot_chain.create_chain()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    chat_history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    reasoning: Dict

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history_msgs = []
    for m in req.chat_history:
        if m.role == "user":
            history_msgs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history_msgs.append(AIMessage(content=m.content))

    db_msgs, reasoning = chatbot_chain.get_database_responses(
        pergunta_usuario=req.user_input,
        chat_history=history_msgs,
        nome_colecao=os.getenv("NOME_COLECAO_DB", "qa_excel_collection"),
        diretorio_db=os.getenv("DIRETORIO_DB"),
    )

    llm_input = {
        "user_input": req.user_input,
        "chat_history": history_msgs,
        "database_responses": db_msgs,
    }
    ai_response = chat_model.invoke(llm_input)
    return ChatResponse(response=ai_response, reasoning=reasoning)
