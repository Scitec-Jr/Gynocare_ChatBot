import os
import json
from typing import Any, List, Tuple

import streamlit as st

# ----------------------------------------------------
# Módulo local contendo o motor de FAQ (garante banco e faz query)
# ----------------------------------------------------
import banco  # noqa: E402

# ====================================================
# Configurações – ajuste via variáveis de ambiente, se quiser
# ====================================================
CAMINHO_PLANILHA = os.getenv("FAQ_XLSX", "PERGUNTAS E RESPOSTAS (SITE).xlsx")
NOME_COLECAO = os.getenv("FAQ_COLECAO", "faq_gynocare")
FORCAR_RECONSTRUCAO = False  # True para recriar a coleção do zero
TOP_K = 5  # quantos resultados retornar

# ====================================================
# Helpers
# ====================================================

IGNORAR_TOKENS = {"idade", "resposta", "n/a", "nan", ""}


def _normalizar_str(v: Any) -> str:
    """Converte para str, remove pipes extras e espaços."""
    return str(v).replace("|", "\\|").strip()


def _get_pares_idade_resposta(metadata: Any) -> List[Tuple[str, str]]:
    """Extrai lista [(idade, resposta), ...] dos metadados do ChromaDB.

    A função aceita duas estruturas geradas pelo *banco.py*:
      • **String JSON** contendo lista de dicts: `[{"idade": "0‑2", "resposta": "..."}, ...]`
      • **Dict** cujo value `"respostas_por_idade_json"` seja a string acima.

    Linhas cujo par (idade ou resposta) esteja em `IGNORAR_TOKENS` são descartadas.
    """
    json_str: str | None = None

    # Metadados já vêm como dict (caso padrão)
    if isinstance(metadata, dict):
        json_str = metadata.get("respostas_por_idade_json")
        # fallback: se algum outro campo for a str JSON
        if json_str is None:
            for v in metadata.values():
                if isinstance(v, str) and v.lstrip().startswith("["):
                    json_str = v
                    break
    # Ou metadados diretamente como string JSON
    elif isinstance(metadata, str):
        json_str = metadata

    if not json_str:
        return []

    try:
        dados = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    pares: List[Tuple[str, str]] = []
    # Estrutura padrão = list[dict]
    if isinstance(dados, list):
        for item in dados:
            if isinstance(item, dict):
                idade_v = _normalizar_str(item.get("idade", ""))
                resp_v = _normalizar_str(item.get("resposta", ""))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                idade_v = _normalizar_str(item[0])
                resp_v = _normalizar_str(item[1])
            else:
                continue

            if idade_v.lower() in IGNORAR_TOKENS and resp_v.lower() in IGNORAR_TOKENS:
                # ignora cabeçalho ou lixo
                continue

            pares.append((idade_v, resp_v))

    return pares


def _montar_tabela_markdown(pares: List[Tuple[str, str]]) -> str:
    """Converte pares em tabela Markdown (ou string fallback)."""
    if not pares:
        return "*Sem respostas cadastradas*"

    linhas = ["| Idade | Resposta |", "| :---- | :------- |"]
    for idade, resp in pares:
        linhas.append(f"| {idade} | {resp} |")
    return "\n".join(linhas)


# ====================================================
# Cache de carregamento do vetor store
# ====================================================
@st.cache_resource(hash_funcs={"chromadb.api.models.Collection.Collection": id})
def carregar_colecao():
    """Cria ou carrega a coleção vetorial do FAQ."""
    return banco.garantir_banco_vetorial_de_xlsx(
        caminho_xlsx=CAMINHO_PLANILHA,
        nome_colecao=NOME_COLECAO,
        forcar_reconstrucao=FORCAR_RECONSTRUCAO,
    )

colecao = carregar_colecao()

# ====================================================
# Interface Chat
# ====================================================

st.title("📚 FAQ Semântico")

if "messages" not in st.session_state:
    st.session_state.messages = []  # histórico

# Renderiza histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=False)

# Entrada
pergunta_usuario = st.chat_input("Digite sua pergunta...")

if pergunta_usuario:
    # Mostra pergunta do user
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})

    # Consulta top‑k
    resultado = colecao.query(
        query_texts=[pergunta_usuario],
        n_results=TOP_K,
        include=["documents", "distances", "metadatas"],
    )

    docs = resultado["documents"][0]
    dists = resultado["distances"][0]
    metas = resultado["metadatas"][0]

    respostas_md: List[str] = []
    for idx, (doc, dist, meta) in enumerate(zip(docs, dists, metas), 1):
        pares = _get_pares_idade_resposta(meta)
        tabela = _montar_tabela_markdown(pares)
        respostas_md.append(
            f"### Resultado {idx}\n"
            f"**Pergunta correspondente:** {doc}\n\n"
            f"{tabela}\n\n"
            f"_Similaridade (cosine distance): {dist:.4f}_"
        )

    resposta_final = "\n\n".join(respostas_md)

    # Exibe resposta
    with st.chat_message("assistant"):
        st.markdown(resposta_final, unsafe_allow_html=False)

    st.session_state.messages.append({"role": "assistant", "content": resposta_final})

# Rodapé
st.markdown("---")
link_repo = "https://github.com/"  # opcional: link para repositório do projeto
st.caption(
    "Aplicação construída com **Streamlit** + **ChromaDB** usando base de FAQs Excel. "
    "Altere `TOP_K` para controlar quantas respostas exibir."
)
