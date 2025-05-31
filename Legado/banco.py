import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os
import json
# hashlib foi importado no código original, mas não usado ativamente.
# Se a ideia fosse implementar IDs estáveis baseados no conteúdo para atualizações incrementais,
# ele seria usado. Por ora, manteremos a importação caso essa funcionalidade seja expandida.
import hashlib

# --- Constantes (Boa prática para valores usados em múltiplos lugares) ---
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHROMA_DIR_PREFIX = "./chroma_db_gynocare_" # Prefixo para diretórios de DB
DEFAULT_METADATA_SPACE = "cosine" # Métrica de similaridade

# --- Função para Criar ou Carregar o Banco de Dados Vetorial ---
def garantir_banco_vetorial_de_xlsx(
    caminho_xlsx: str,
    nome_colecao: str = "qa_excel_collection",
    diretorio_db: str | None = None, # Permitir None para usar um padrão
    forcar_reconstrucao: bool = False
) -> chromadb.Collection | None:
    """
    Garante que uma coleção ChromaDB exista e esteja populada a partir de um arquivo XLSX.
    Se a coleção não existir ou 'forcar_reconstrucao' for True, ela será (re)criada
    a partir do arquivo XLSX. Caso contrário, a coleção existente é carregada.

    A estrutura esperada do XLSX é:
    Primeira coluna: Perguntas (pode ter células mescladas)
    Segunda coluna:  Idade/Contexto específico
    Terceira coluna: Resposta

    Args:
        caminho_xlsx: Caminho para o arquivo .xlsx. Necessário se a coleção
                      precisar ser criada/reconstruída.
        nome_colecao: Nome para a coleção no ChromaDB.
        diretorio_db: Diretório para persistir o ChromaDB. Se None, um padrão
                      baseado no nome da coleção será usado.
        forcar_reconstrucao: Se True, deleta e recria a coleção a partir do XLSX,
                             mesmo que ela já exista.

    Returns:
        O objeto da coleção ChromaDB ou None se houver erro.
    """
    # Define o diretório de persistência padrão se não for fornecido
    if diretorio_db is None:
        diretorio_db = f"{DEFAULT_CHROMA_DIR_PREFIX}{nome_colecao}"

    # Garante que o diretório de persistência exista
    if not os.path.exists(diretorio_db):
        try:
            os.makedirs(diretorio_db)
            print(f"Diretório de persistência criado: {diretorio_db}")
        except OSError as e:
            print(f"Erro ao criar o diretório de persistência '{diretorio_db}': {e}")
            return None
    
    # Inicializa o cliente ChromaDB persistente
    try:
        client = chromadb.PersistentClient(path=diretorio_db)
    except Exception as e:
        print(f"Erro ao inicializar o cliente ChromaDB em '{diretorio_db}': {e}")
        return None

    # Configura a função de embedding
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=DEFAULT_MODEL_NAME)

    colecao_existente = None
    # Tenta obter a coleção se ela já existir
    if not forcar_reconstrucao: # Só tenta obter se não for forçar reconstrução
        try:
            colecao_existente = client.get_collection(name=nome_colecao, embedding_function=embedding_function)
            print(f"Coleção '{nome_colecao}' encontrada e carregada de '{diretorio_db}'.")
            return colecao_existente # Retorna a coleção existente
        except Exception: # Captura exceções como CollectionNotFound, etc.
            print(f"Coleção '{nome_colecao}' não encontrada em '{diretorio_db}'. Será criada.")
            pass # colecao_existente permanece None, o que levará à criação

    # Se forçar reconstrução e a coleção existir, deleta primeiro
    if forcar_reconstrucao:
        try:
            # Verifica se existe antes de tentar deletar para evitar log de erro desnecessário
            client.get_collection(name=nome_colecao) # Isso lança exceção se não existir
            print(f"Forçando reconstrução: Deletando coleção '{nome_colecao}' existente...")
            client.delete_collection(name=nome_colecao)
        except Exception:
            # Se get_collection falhou (não existe), não há nada para deletar.
            # Ou se delete_collection falhou por algum motivo (raro se get_collection funcionou).
            print(f"Nota: Coleção '{nome_colecao}' não existia para ser deletada ou ocorreu um erro na tentativa de deleção.")
            pass

    # --- Processamento do Arquivo XLSX para criar/recriar a coleção ---
    print(f"Processando arquivo XLSX '{caminho_xlsx}' para (re)criar a coleção '{nome_colecao}'...")
    try:
        # Lê o arquivo XLSX, assumindo que a primeira linha é o cabeçalho
        df = pd.read_excel(caminho_xlsx, header=0)
        
        # Verifica se o DataFrame tem o número mínimo de colunas esperado
        if df.shape[1] < 3:
            print(f"Erro: A planilha '{caminho_xlsx}' precisa ter pelo menos 3 colunas. Encontradas: {df.shape[1]}.")
            return None
        
        # Seleciona apenas as três primeiras colunas e as renomeia para padronização
        df = df.iloc[:, :3]
        df.columns = ['pergunta', 'idade', 'resposta']
        print(f"Planilha lida e colunas renomeadas. {df.shape[0]} linhas para processar.")

    except FileNotFoundError:
        print(f"Erro: Arquivo XLSX não encontrado em '{caminho_xlsx}'.")
        return None
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo XLSX '{caminho_xlsx}': {e}")
        return None

    # Preenche os valores de 'pergunta' para baixo para lidar com células mescladas
    # O pandas lê células mescladas colocando o valor na primeira linha e NaN nas subsequentes.
    # ffill() propaga o último valor válido para preencher os NaNs.
    df['pergunta'] = df['pergunta'].ffill()

    # Agrupa os dados: cada pergunta única terá uma lista de pares {'idade': valor, 'resposta': valor}
    perguntas_agrupadas = {}
    for index, row in df.iterrows():
        pergunta_original = str(row['pergunta']).strip()
        
        # Trata valores NaN para idade e resposta, substituindo por "N/A" para consistência
        idade_valor = "N/A" if pd.isna(row['idade']) else str(row['idade']).strip()
        resposta_valor = "N/A" if pd.isna(row['resposta']) else str(row['resposta']).strip()

        # Ignora linhas se a pergunta original for vazia, NaN ou a string "nan"
        if not pergunta_original or pergunta_original.lower() == 'nan' or pd.isna(row['pergunta']):
            # print(f"Debug: Pulando linha {index+2} do Excel devido à pergunta inválida: '{row['pergunta']}'")
            continue
        
        # Inicializa a lista de respostas para uma nova pergunta
        if pergunta_original not in perguntas_agrupadas:
            perguntas_agrupadas[pergunta_original] = []
        
        # Adiciona o par idade/resposta à lista da pergunta correspondente
        # Mesmo que idade ou resposta sejam "N/A", elas são adicionadas se a pergunta for válida.
        # A lógica de apresentação na busca pode decidir como exibir "N/A".
        perguntas_agrupadas[pergunta_original].append({"idade": idade_valor, "resposta": resposta_valor})

    if not perguntas_agrupadas:
        print("Nenhuma pergunta válida encontrada na planilha após o agrupamento.")
        return None

    # --- Criação da Coleção no ChromaDB ---
    try:
        collection = client.create_collection(
            name=nome_colecao,
            embedding_function=embedding_function,
            metadata={"hnsw:space": DEFAULT_METADATA_SPACE} # Usando constante
        )
        print(f"Coleção '{nome_colecao}' criada com sucesso.")
    except chromadb.errors.DuplicateCollectionException:
        print(f"Aviso: Tentativa de criar coleção '{nome_colecao}' que já existe (deleção pode ter falhado ou condição de corrida). Tentando obter...")
        try:
            collection = client.get_collection(name=nome_colecao, embedding_function=embedding_function)
            print(f"Sucesso ao obter a coleção '{nome_colecao}' que já existia.")
        except Exception as e_get:
            print(f"Falha crítica: Não foi possível criar nem obter a coleção '{nome_colecao}'. Erro: {e_get}")
            return None
    except Exception as e:
        print(f"Erro inesperado ao criar a coleção '{nome_colecao}': {e}")
        return None

    # --- População da Coleção com Dados Processados ---
    ids_para_chroma = []
    documentos_para_embedding = [] # As perguntas únicas serão embutidas
    metadados_para_chroma = []
    item_count = 0

    for pergunta_unica, lista_de_respostas_associadas in perguntas_agrupadas.items():
        # Pula se uma pergunta acabou sem respostas válidas associadas (raro com a lógica atual, mas seguro)
        if not lista_de_respostas_associadas:
            continue

        # Cria um ID para o item no ChromaDB
        # Usar um hash da pergunta tornaria o ID estável e útil para atualizações incrementais,
        # mas para o modo de reconstrução, um contador simples é suficiente.
        # id_item = hashlib.md5(pergunta_unica.encode()).hexdigest() # Exemplo de ID estável
        id_item = f"q_excel_{item_count}"
        ids_para_chroma.append(id_item)
        
        # O documento a ser embutido é a própria pergunta única
        documentos_para_embedding.append(pergunta_unica)
        
        # Serializa a lista de respostas associadas para uma string JSON
        # Isso é necessário porque o ChromaDB (com DuckDB backend) espera valores escalares nos metadados.
        respostas_json_string = json.dumps(lista_de_respostas_associadas)
        
        metadados_para_chroma.append({
            "pergunta_original_excel": pergunta_unica,      # Para referência e exibição
            "respostas_por_idade_json": respostas_json_string # A estrutura complexa armazenada como JSON
        })
        item_count += 1

    # Adiciona os dados processados à coleção ChromaDB
    if documentos_para_embedding:
        try:
            collection.add(
                ids=ids_para_chroma,
                documents=documentos_para_embedding,
                metadatas=metadados_para_chroma
            )
            print(f"{len(documentos_para_embedding)} perguntas únicas foram adicionadas/embutidas na coleção '{nome_colecao}'.")
        except Exception as e:
            print(f"Erro ao adicionar dados à coleção '{nome_colecao}': {e}")
            # Considerar deletar a coleção se a adição falhar para evitar estado inconsistente
            # client.delete_collection(name=nome_colecao)
            return None # Ou retornar a coleção parcialmente populada, dependendo da política
    else:
        print("Nenhum documento foi preparado para adicionar à coleção (após processamento do XLSX).")
        # Se a coleção foi criada, mas nada será adicionado, pode ser considerada "vazia".
        # Pode-se optar por deletá-la ou retorná-la assim mesmo.
        # client.delete_collection(name=nome_colecao)
        # return None

    return collection


# --- Função para Buscar no Banco de Dados e Formatar Saída ---
def buscar_pergunta_no_banco(
    pergunta_usuario: str,
    nome_colecao: str = "qa_excel_collection",
    diretorio_db: str | None = None
) -> tuple[str | None, str | None, float | None]:
    """
    Busca uma pergunta do usuário no banco de Dados ChromaDB especificado.
    Retorna a pergunta mais similar encontrada na base, uma tabela Markdown
    das idades/respostas associadas, e a distância de similaridade.

    Args:
        pergunta_usuario: A pergunta feita pelo usuário.
        nome_colecao: Nome da coleção no ChromaDB.
        diretorio_db: Diretório onde o ChromaDB está persistido. Se None,
                      um padrão baseado no nome da coleção será usado.

    Returns:
        Uma tupla: (pergunta_da_base, tabela_markdown_respostas, distancia_similaridade).
        Retorna (None, None, None) se nenhum resultado for encontrado ou em caso de erro.
    """
    if diretorio_db is None:
        diretorio_db = f"{DEFAULT_CHROMA_DIR_PREFIX}{nome_colecao}"

    if not os.path.exists(diretorio_db):
        print(f"Erro: Diretório do banco de dados '{diretorio_db}' não encontrado. A coleção '{nome_colecao}' não pode ser carregada.")
        return None, None, None

    try:
        client = chromadb.PersistentClient(path=diretorio_db)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=DEFAULT_MODEL_NAME)
        collection = client.get_collection(name=nome_colecao, embedding_function=embedding_function)
    except Exception as e:
        print(f"Erro ao carregar a coleção '{nome_colecao}' de '{diretorio_db}': {e}")
        print("  Verifique se o banco de dados foi criado corretamente e se o nome da coleção e o diretório estão corretos.")
        return None, None, None

    # Realiza a query na coleção
    try:
        query_results = collection.query(
            query_texts=[pergunta_usuario],
            n_results=1, # Pega apenas o resultado mais similar
            include=['metadatas', 'distances', 'documents'] # Inclui os dados necessários
        )
    except Exception as e:
        print(f"Erro durante a execução da query na coleção '{nome_colecao}': {e}")
        return None, None, None

    # Processa os resultados da query
    if query_results['ids'] and query_results['ids'][0]: # Verifica se há resultados
        # O primeiro [0] é para a primeira (e única) query_text, o segundo [0] é para o primeiro (n_results=1) resultado.
        best_match_metadata = query_results['metadatas'][0][0]
        best_match_distance = query_results['distances'][0][0]
        # O 'document' armazenado é a pergunta original do Excel que foi embutida
        retrieved_excel_question = query_results['documents'][0][0]

        # Desserializa a string JSON contendo os pares idade/resposta
        respostas_json_string = best_match_metadata.get("respostas_por_idade_json")
        respostas_por_idade = []
        if respostas_json_string:
            try:
                respostas_por_idade = json.loads(respostas_json_string)
            except json.JSONDecodeError as e:
                print(f"Erro ao desserializar JSON dos metadados para a pergunta '{retrieved_excel_question}': {e}")
                # Trata como se não houvesse respostas válidas em caso de erro de desserialização
                respostas_por_idade = [] 
        
        # Verificação de consistência dos dados recuperados
        if not retrieved_excel_question or not isinstance(respostas_por_idade, list):
            print(f"Aviso: Dados recuperados para a pergunta '{retrieved_excel_question or 'DESCONHECIDA'}' parecem incompletos ou malformados nos metadatos.")
            # Decide se retorna a pergunta com uma mensagem de erro ou None total
            # Aqui, vamos retornar a pergunta e uma tabela indicando o problema.
            markdown_table_error = "| Idade    | Resposta Correspondente     |\n"
            markdown_table_error += "| :------- | :-------------------------- |\n"
            markdown_table_error += "| ERRO     | Dados de resposta inválidos |\n"
            return retrieved_excel_question, markdown_table_error.strip(), best_match_distance


        # Cria a tabela Markdown para as respostas
        markdown_table_parts = [
            "| Idade    | Resposta Correspondente |", # Cabeçalho
            "| :------- | :---------------------- |"  # Separador com alinhamento
        ]
        
        if not respostas_por_idade: # Se a lista de respostas estiver vazia
            markdown_table_parts.append("| N/A      | Nenhuma resposta específica encontrada |")
        else:
            for item in respostas_por_idade:
                # Escapa o caractere pipe '|' para não quebrar a formatação da tabela Markdown
                idade_escaped = str(item.get('idade', 'N/A')).replace('|', '\\|')
                resposta_escaped = str(item.get('resposta', 'N/A')).replace('|', '\\|')
                markdown_table_parts.append(f"| {idade_escaped} | {resposta_escaped} |")
        
        tabela_markdown_final = "\n".join(markdown_table_parts)

        return retrieved_excel_question, tabela_markdown_final, best_match_distance
    else:
        # Nenhum resultado similar encontrado pela query
        return None, None, None


# --- Exemplo de Uso Principal ---
if __name__ == "__main__":
    # Configurações para o seu ambiente e arquivo
    caminho_da_sua_planilha = "./PERGUNTAS E RESPOSTAS (SITE).xlsx"
    nome_da_colecao_db = "base_gynocare_final" # Use um nome descritivo
    diretorio_de_persistencia_db = f"{DEFAULT_CHROMA_DIR_PREFIX}{nome_da_colecao_db}" # Gera o diretório com base no nome

    # --- Etapa 1: Garantir que o Banco de Dados Vetorial exista e esteja populado ---
    print(f"--- INICIALIZAÇÃO DO BANCO DE DADOS VETORIAL ---")
    # Verifique se o arquivo da planilha existe ANTES de chamar a função de criação/garantia
    if not os.path.exists(caminho_da_sua_planilha):
        print(f"ERRO CRÍTICO: A planilha '{caminho_da_sua_planilha}' não foi encontrada.")
        print("  O script não pode continuar sem o arquivo de dados.")
        exit(1) # Sai do script com código de erro
    else:
        print(f"Utilizando planilha de dados de: '{caminho_da_sua_planilha}'")

    # Chame para criar o DB na primeira vez, ou carregar se já existir.
    # Defina forcar_reconstrucao=True se quiser recriar o DB a partir do Excel,
    # por exemplo, se o arquivo Excel foi atualizado.
    colecao_app = garantir_banco_vetorial_de_xlsx(
        caminho_xlsx=caminho_da_sua_planilha,
        nome_colecao=nome_da_colecao_db,
        diretorio_db=diretorio_de_persistencia_db,
        forcar_reconstrucao=False # Mude para True para forçar a recriação do zero
    )

    if not colecao_app:
        print("ERRO CRÍTICO: Falha ao criar ou carregar o banco de dados vetorial.")
        print("  Verifique os logs de erro acima para mais detalhes.")
        exit(1) # Sai do script com código de erro
    
    print(f"Banco de dados vetorial '{colecao_app.name}' está pronto com {colecao_app.count()} perguntas únicas embutidas.")
    print("-" * 50)

    # --- Etapa 2: Realizar Buscas no Banco de Dados ---
    print(f"\n--- SISTEMA DE BUSCA DE PERGUNTAS ---")
    perguntas_para_teste = [
        "Quais exames da parte de Pediatria são realizados?",
        "Qual preparo para a ecografia pélvica",
        "O que é a dor no rim?",
        "Posso me atrasar?",
        "Meu irmão pode ir comigo?",
        "Como funciona o agendamento online?", # Exemplo de pergunta que pode não estar na base
        "Informações sobre vacinação infantil"
    ]

    for pergunta_teste_usuario in perguntas_para_teste:
        print(f"\nUsuário pergunta: \"{pergunta_teste_usuario}\"")

        pergunta_encontrada, tabela_markdown, distancia_sim = buscar_pergunta_no_banco(
            pergunta_usuario=pergunta_teste_usuario,
            nome_colecao=nome_da_colecao_db,
            diretorio_db=diretorio_de_persistencia_db
        )

        print("\n[RESULTADO DA BUSCA]")
        if pergunta_encontrada:
            print(f"Pergunta Mais Similar Encontrada na Base de Dados (Distância: {distancia_sim:.4f}):")
            print(f"  -> \"{pergunta_encontrada}\"")
            print(f"\nIdades/Contextos e Respostas Associadas:")
            print(tabela_markdown)
        else:
            print("Nenhuma pergunta/resposta correspondente encontrada na base de dados para esta pergunta.")
        print("-" * 40)

    print("\n--- FIM DAS BUSCAS DE EXEMPLO ---")