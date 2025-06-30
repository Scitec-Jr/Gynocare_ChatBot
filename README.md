# Gynocare ChatBot

Este repositório agora é composto por dois componentes principais:

1. **API (FastAPI)** - arquivo `api.py`
2. **Aplicativo Streamlit** - arquivo `app.py`

O aplicativo Streamlit não processa mais o modelo localmente. Ele apenas envia as mensagens para a API e exibe a resposta.

## Como hospedar a API gratuitamente

Uma opção simples é utilizar o [Railway](https://railway.app) ou o [Render](https://render.com). Ambos possuem planos gratuitos que permitem executar uma API Python.

Passos básicos (exemplo com Railway):

1. Crie uma conta gratuita em `railway.app`.
2. Inicie um novo projeto e conecte este repositório ou faça upload dos arquivos.
3. Defina o comando de start para `uvicorn api:app --host 0.0.0.0 --port $PORT`.
4. Adicione as variáveis de ambiente necessárias (por exemplo, `GROQ_API_KEY`, `NOME_COLECAO_DB` etc.).
5. Deploy e aguarde o Railway disponibilizar a URL.

Com a API rodando, defina a variável `API_URL` do Streamlit para apontar para essa URL.

No Streamlit Cloud (ou outro provedor), mantenha apenas `app.py`. Ele enviará as mensagens para a sua API hospedada.

