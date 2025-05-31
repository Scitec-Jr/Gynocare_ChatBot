import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# --- Constantes (valores reutilizados em vários lugares) ---
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIR_PREFIX = "./chroma_db_gynocare_"
DEFAULT_METADATA_SPACE = "cosine"


def garantir_banco_vetorial_de_xlsx(
    caminho_xlsx: str,
    nome_colecao: str = "qa_excel_collection",
    diretorio_db: str | None = None,
    forcar_reconstrucao: bool = False
) -> chromadb.Collection | None:
    """
    Garante que uma coleção ChromaDB exista e esteja populada a partir de um arquivo XLSX.
    Se a coleção não existir ou 'forcar_reconstrucao' for True, ela será (re)criada a partir do arquivo XLSX.

    Args:
        caminho_xlsx: Caminho para o arquivo .xlsx. Necessário se a coleção precisar ser criada/reconstruída.
        nome_colecao: Nome para a coleção no ChromaDB.
        diretorio_db: Diretório para persistir o ChromaDB. Se None, um padrão baseado no nome da coleção será usado.
        forcar_reconstrucao: Se True, deleta e recria a coleção a partir do XLSX.

    Returns:
        A coleção ChromaDB ou None em caso de erro.
    """
    if diretorio_db is None:
        diretorio_db = f"{DEFAULT_CHROMA_DIR_PREFIX}{nome_colecao}"

    # Cria pasta de persistência, se não existir
    if not os.path.exists(diretorio_db):
        try:
            os.makedirs(diretorio_db)
        except OSError as e:
            print(f"Erro ao criar o diretório de persistência '{diretorio_db}': {e}")
            return None

    # Inicializa o cliente ChromaDB persistente
    try:
        client = chromadb.PersistentClient(path=diretorio_db)
    except Exception as e:
        print(f"Erro ao inicializar o cliente ChromaDB em '{diretorio_db}': {e}")
        return None

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=DEFAULT_MODEL_NAME
    )

    # Se não for forçar reconstrução, tenta recuperar a coleção existente
    if not forcar_reconstrucao:
        try:
            colecao_existente = client.get_collection(
                name=nome_colecao,
                embedding_function=embedding_function
            )
            return colecao_existente
        except Exception:
            pass  # Se não existir, vamos criar abaixo

    # Se for forçar reconstrução, tenta deletar a coleção antiga antes de recriar
    if forcar_reconstrucao:
        try:
            client.get_collection(name=nome_colecao)
            client.delete_collection(name=nome_colecao)
        except Exception:
            pass

    # Lê o XLSX e monta um DataFrame com colunas: ['pergunta', 'idade', 'resposta']
    try:
        df = pd.read_excel(caminho_xlsx, header=0)
        if df.shape[1] < 3:
            print(f"Erro: A planilha '{caminho_xlsx}' precisa ter pelo menos 3 colunas. Encontradas: {df.shape[1]}.")
            return None
        df = df.iloc[:, :3]
        df.columns = ["pergunta", "idade", "resposta"]
    except FileNotFoundError:
        print(f"Erro: Arquivo XLSX não encontrado em '{caminho_xlsx}'.")
        return None
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo XLSX '{caminho_xlsx}': {e}")
        return None

    # Preenche perguntas em branco com o valor acima (caso a planilha use merges)
    df["pergunta"] = df["pergunta"].ffill()

    # Agrupa respostas por pergunta
    perguntas_agrupadas: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        pergunta_orig = str(row["pergunta"]).strip()
        if not pergunta_orig or pergunta_orig.lower() == "nan":
            continue

        idade_valor = "N/A" if pd.isna(row["idade"]) else str(row["idade"]).strip()
        resposta_valor = "N/A" if pd.isna(row["resposta"]) else str(row["resposta"]).strip()

        if pergunta_orig not in perguntas_agrupadas:
            perguntas_agrupadas[pergunta_orig] = []
        perguntas_agrupadas[pergunta_orig].append({"idade": idade_valor, "resposta": resposta_valor})

    if not perguntas_agrupadas:
        print("Nenhuma pergunta válida encontrada na planilha.")
        return None

    # Cria a coleção (ou obtém se já existir)
    try:
        collection = client.create_collection(
            name=nome_colecao,
            embedding_function=embedding_function,
            metadata={"hnsw:space": DEFAULT_METADATA_SPACE}
        )
    except chromadb.errors.DuplicateCollectionException:
        try:
            collection = client.get_collection(
                name=nome_colecao,
                embedding_function=embedding_function
            )
        except Exception as e_get:
            print(f"Falha ao obter a coleção existente '{nome_colecao}': {e_get}")
            return None
    except Exception as e:
        print(f"Erro inesperado ao criar a coleção '{nome_colecao}': {e}")
        return None

    ids_para_chroma: list[str] = []
    documentos_para_embedding: list[str] = []
    metadados_para_chroma: list[dict[str, str]] = []
    item_count = 0

    # Para cada pergunta única, criamos um ID, armazenamos a pergunta para embedding
    # e armazenamos no metadata o JSON com todas as faixas etárias e respostas
    for pergunta_unica, respostas_list in perguntas_agrupadas.items():
        if not respostas_list:
            continue

        id_item = f"q_excel_{item_count}"
        ids_para_chroma.append(id_item)
        documentos_para_embedding.append(pergunta_unica)
        respostas_json = json.dumps(respostas_list)
        metadados_para_chroma.append({
            "pergunta_original_excel": pergunta_unica,
            "respostas_por_idade_json": respostas_json
        })
        item_count += 1

    if documentos_para_embedding:
        try:
            collection.add(
                ids=ids_para_chroma,
                documents=documentos_para_embedding,
                metadatas=metadados_para_chroma
            )
        except Exception as e:
            print(f"Erro ao adicionar dados à coleção '{nome_colecao}': {e}")
            return None
    else:
        return None

    return collection


def buscar_perguntas_no_banco(
    pergunta_usuario: str,
    nome_colecao: str = "qa_excel_collection",
    diretorio_db: str | None = None,
    n_results: int = 3
) -> list[tuple[str, str, float]]:
    """
    Busca as n perguntas mais similares no banco de dados ChromaDB.

    Args:
        pergunta_usuario: A pergunta feita pelo usuário.
        nome_colecao: Nome da coleção no ChromaDB.
        diretorio_db: Diretório onde o ChromaDB está persistido. Se None, usa padrão.
        n_results: Número de resultados mais similares a retornar.

    Returns:
        Lista de tuplas: [(pergunta_base, tabela_markdown, distancia), ...].
        Retorna lista vazia se não houver resultados ou em caso de erro.
    """
    if diretorio_db is None:
        diretorio_db = f"{DEFAULT_CHROMA_DIR_PREFIX}{nome_colecao}"

    if not os.path.exists(diretorio_db):
        return []

    try:
        client = chromadb.PersistentClient(path=diretorio_db)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_MODEL_NAME
        )
        collection = client.get_collection(
            name=nome_colecao,
            embedding_function=embedding_function
        )
    except Exception:
        return []

    try:
        query_results = collection.query(
            query_texts=[pergunta_usuario],
            n_results=n_results,
            include=["metadatas", "distances", "documents"]
        )
    except Exception:
        return []

    resultados: list[tuple[str, str, float]] = []

    # Se vierem IDs, montamos a tabela de idade x resposta para cada match
    if query_results["ids"] and query_results["ids"][0]:
        for idx in range(len(query_results["ids"][0])):
            metadata = query_results["metadatas"][0][idx]
            distancia = query_results["distances"][0][idx]
            pergunta_base = query_results["documents"][0][idx]

            respostas_json = metadata.get("respostas_por_idade_json", "[]")
            try:
                respostas_por_idade = json.loads(respostas_json)
            except json.JSONDecodeError:
                respostas_por_idade = []

            markdown_table = [
                "| Idade    | Resposta Correspondente |",
                "| :------- | :---------------------- |"
            ]

            if not respostas_por_idade:
                markdown_table.append("| N/A      | Nenhuma resposta encontrada |")
            else:
                for item in respostas_por_idade:
                    idade_esc = str(item.get("idade", "N/A")).replace("|", "\\|")
                    resp_esc = str(item.get("resposta", "N/A")).replace("|", "\\|")
                    markdown_table.append(f"| {idade_esc} | {resp_esc} |")

            tabela_final = "\n".join(markdown_table)
            resultados.append((pergunta_base, tabela_final, distancia))

    return resultados

garantir_banco_vetorial_de_xlsx(
    caminho_xlsx="PERGUNTAS E RESPOSTAS (SITE).xlsx",
    nome_colecao="qa_excel_collection",
    diretorio_db="./chroma_db_gynocare_qa_excel_collection",
    forcar_reconstrucao=False
)