import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


# Função para conectar ao banco de dados SQLite
def conectar_banco(nome_db=r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\database\index_arquivos.db"):
    try:
        conn = sqlite3.connect(nome_db)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None, None


# Função para carregar os embeddings do banco de dados
def carregar_embeddings(cursor):
    try:
        cursor.execute("SELECT id, embedding FROM conteudo")
        registros = cursor.fetchall()
        ids = []
        embeddings = []

        for id_conteudo, embedding_blob in registros:
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                # Verifica se há NaNs ou infinitos nos embeddings
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    print(f"Embedding ID {id_conteudo} contém NaN ou Inf. Ignorando...")
                    continue

                embeddings.append(embedding)
                ids.append(id_conteudo)

        return ids, np.array(embeddings)
    except Exception as e:
        print(f"Erro ao carregar embeddings: {e}")
        return [], np.array([])


# Função para verificar proximidade entre embeddings
def verificar_proximidade(embeddings, ids, threshold=0.1):
    """
    Verifica se os embeddings estão muito próximos usando a distância cosseno.
    Args:
        embeddings (np.ndarray): Array de embeddings.
        ids (list): Lista de IDs correspondentes aos embeddings.
        threshold (float): Limite de distância cosseno para considerar "próximo".
    Returns:
        list: Lista de pares de IDs com embeddings muito próximos.
    """
    proximos = []
    num_embeddings = len(embeddings)

    # Calcula a matriz de distâncias cosseno
    distancias = cosine_distances(embeddings)

    # Verifica cada par de embeddings
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            if distancias[i, j] < threshold:
                proximos.append((ids[i], ids[j], distancias[i, j]))

    return proximos


# Função principal
def main():
    # Conecta ao banco de dados
    conn, cursor = conectar_banco()
    if conn is None or cursor is None:
        print("Não foi possível conectar ao banco de dados.")
        return

    # Carrega os embeddings do banco de dados
    ids, embeddings = carregar_embeddings(cursor)
    if len(embeddings) == 0:
        print("Nenhum embedding válido encontrado no banco de dados.")
        conn.close()
        return

    # Define o limite de proximidade (threshold)
    threshold = 0.1  # Ajuste conforme necessário

    # Verifica se há embeddings muito próximos
    pares_proximos = verificar_proximidade(embeddings, ids, threshold)

    # Exibe os resultados
    if len(pares_proximos) > 0:
        print("\nAtenção: Foram encontrados embeddings muito próximos!")
        print("Pares de IDs com distâncias abaixo do limite:")
        for id1, id2, distancia in pares_proximos:
            print(f"ID {id1} e ID {id2}: Distância = {distancia:.4f}")
    else:
        print("\nNenhum par de embeddings muito próximo foi encontrado.")

    # Fecha a conexão com o banco de dados
    conn.close()


# Executa o programa
if __name__ == "__main__":
    main()