import sqlite3
import numpy as np

def verificar_embeddings(nome_db=r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\database\index_arquivos.db"):
    try:
        # Conectar ao banco de dados
        conn = sqlite3.connect(nome_db)
        cursor = conn.cursor()

        # Lista de colunas de embeddings na tabela "embeddings"
        colunas_embeddings = [
            "embedding_nome", "embedding_caminho", "embedding_tipo",
            "embedding_tamanho", "embedding_data", "embedding_status",
            "embedding_conteudo"
        ]

        total_embeddings = 0
        embeddings_validos = 0

        print("\nVerificando embeddings salvos no banco de dados...\n")

        for coluna in colunas_embeddings:
            cursor.execute(f"SELECT id, {coluna} FROM embeddings")
            registros = cursor.fetchall()

            total_embeddings += len(registros)

            for id_arquivo, embedding_blob in registros:
                if embedding_blob is None or len(embedding_blob) == 0:
                    print(f"Erro: Embedding '{coluna}' do ID {id_arquivo} está vazio ou NULL!")
                    continue

                try:
                    # Converter o BLOB para um array NumPy
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                    # Verifica se o tamanho do embedding está correto
                    if embedding.shape[0] != 768:
                        print(f"Erro: Embedding '{coluna}' do ID {id_arquivo} tem tamanho incorreto ({embedding.shape[0]} em vez de 768)!")
                        continue

                    # Verifica se há NaN ou infinito no embedding
                    if np.isnan(embedding).any() or np.isinf(embedding).any():
                        print(f"Erro: Embedding '{coluna}' do ID {id_arquivo} contém valores inválidos (NaN ou Inf)!")
                        continue

                    # Verifica se a normalização está correta
                    norma = np.linalg.norm(embedding)
                    if not (0.99 <= norma <= 1.01):
                        print(f"Aviso: Embedding '{coluna}' do ID {id_arquivo} tem norma incorreta ({norma:.5f})!")

                    # Imprime os primeiros valores do embedding como amostra
                    print(f"Embedding '{coluna}' ID {id_arquivo}: {embedding[:5]} ... (norma: {norma:.5f})")

                    embeddings_validos += 1

                except Exception as e:
                    print(f"Erro ao processar embedding '{coluna}' do ID {id_arquivo}: {e}")

        conn.close()

        print("\nResumo da verificação:")
        print(f"{embeddings_validos}/{total_embeddings} embeddings verificados corretamente.")

        if embeddings_validos == total_embeddings:
            print("\nTodos os embeddings estão corretos!\n")
        else:
            print("\nAlguns embeddings apresentaram problemas. Recomenda-se reindexar o banco de dados.\n")

    except Exception as e:
        print(f"Erro ao acessar o banco de dados: {e}")

# Executar a verificação
if __name__ == "__main__":
    verificar_embeddings()
