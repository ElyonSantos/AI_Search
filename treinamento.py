import os
import sqlite3
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime


def treinar_modelo_com_db(nome_db=r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\database\index_arquivos.db",
                          modelo_saida="modelo/modelo_chatbot.pkl"):
    # Criar a pasta "modelo" se ela não existir
    modelo_dir = os.path.dirname(modelo_saida)
    if not os.path.exists(modelo_dir):
        os.makedirs(modelo_dir)
        print(f"Pasta '{modelo_dir}' criada!")

    if not os.path.exists(nome_db):
        print(f"Erro: O arquivo '{nome_db}' não foi encontrado.")
        return

    try:
        with sqlite3.connect(nome_db) as conn:
            cursor = conn.cursor()
            print("Carregando dados do banco de dados...")

            # Buscar os dados da tabela arquivos e embeddings
            cursor.execute("""
                SELECT 
                    a.id, a.tipo, a.tamanho, a.data_modificacao, a.status,
                    e.embedding_nome, e.embedding_caminho, e.embedding_tipo, 
                    e.embedding_tamanho, e.embedding_data, e.embedding_status, e.embedding_conteudo
                FROM arquivos a
                LEFT JOIN embeddings e ON a.id = e.id
            """)

            dados = cursor.fetchall()

            if not dados:
                print("Nenhum dado encontrado no banco de dados.")
                return

            ids_arquivos = []
            atributos_numericos = []
            atributos_categoricos = []
            embeddings_lista = {
                "embedding_nome": [],
                "embedding_caminho": [],
                "embedding_tipo": [],
                "embedding_tamanho": [],
                "embedding_data": [],
                "embedding_status": [],
                "embedding_conteudo": []
            }

            for linha in dados:
                id = linha[0]
                ids_arquivos.append(id)

                tipo = linha[1] or "desconhecido"
                tamanho = linha[2] if linha[2] is not None else 0
                data_modificacao = linha[3] or "1970-01-01T00:00:00"
                status = linha[4] or "desconhecido"

                try:
                    timestamp = datetime.fromisoformat(data_modificacao).timestamp()
                except (ValueError, TypeError):
                    timestamp = 0

                atributos_numericos.append([tamanho, timestamp])
                atributos_categoricos.append([tipo, status])

                for i, coluna in enumerate(embeddings_lista.keys(), start=5):
                    embedding_blob = linha[i]
                    if embedding_blob:
                        try:
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(-1)

                            if embedding.shape[0] != 768:
                                print(
                                    f"Erro: Embedding '{coluna}' do ID {id} tem {embedding.shape[0]} dimensões em vez de 768. Corrigindo...")
                                embedding = np.zeros(768)

                            elif np.isnan(embedding).any():
                                print(
                                    f"Erro: Embedding '{coluna}' do ID {id} contém valores inválidos! Reindexe o banco de dados.")
                                embedding = np.zeros(768)

                        except Exception as e:
                            print(f"Erro ao processar embedding '{coluna}' do ID {id}: {e}")
                            embedding = np.zeros(768)
                    else:
                        embedding = np.zeros(768)

                    embeddings_lista[coluna].append(embedding)

            atributos_numericos = np.array(atributos_numericos)
            embeddings_np = {chave: np.array(valor) for chave, valor in embeddings_lista.items()}

            scaler = StandardScaler()
            atributos_numericos_normalizados = scaler.fit_transform(atributos_numericos) if atributos_numericos.shape[
                                                                                                0] > 0 else np.zeros(
                (0, 2))

            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            atributos_categoricos_codificados = encoder.fit_transform(atributos_categoricos) if \
            np.array(atributos_categoricos).shape[0] > 0 else np.zeros((0, 2))

            num_amostras = atributos_numericos.shape[0]

            if atributos_numericos_normalizados.shape[0] != num_amostras:
                atributos_numericos_normalizados = np.zeros((num_amostras, 2))

            if atributos_categoricos_codificados.shape[0] != num_amostras:
                atributos_categoricos_codificados = np.zeros((num_amostras, 2))

            for chave in embeddings_np.keys():
                if embeddings_np[chave].shape[0] != num_amostras:
                    embeddings_np[chave] = np.zeros((num_amostras, 768))

            representacoes_combinadas = np.hstack([
                atributos_numericos_normalizados,
                atributos_categoricos_codificados,
                embeddings_np["embedding_nome"],
                embeddings_np["embedding_caminho"],
                embeddings_np["embedding_tipo"],
                embeddings_np["embedding_tamanho"],
                embeddings_np["embedding_data"],
                embeddings_np["embedding_status"],
                embeddings_np["embedding_conteudo"]
            ])

            if len(representacoes_combinadas) == 0:
                print("Erro: Nenhuma amostra válida encontrada para treinamento!")
                return

            print(f"Quantidade de arquivos identificados: {len(ids_arquivos)}")
            print(
                f"Treinando modelo com {len(representacoes_combinadas)} amostras e {representacoes_combinadas.shape[1]} features...")
            modelo = NearestNeighbors(n_neighbors=5, metric="cosine")
            modelo.fit(representacoes_combinadas)

            if not ids_arquivos:
                print(
                    "Erro: Nenhum ID de arquivo foi armazenado! Verifique se os arquivos foram indexados corretamente.")
                return

            try:
                if not ids_arquivos or len(representacoes_combinadas) == 0:
                    raise ValueError("O modelo não tem amostras suficientes para treinamento!")

                with open(modelo_saida, "wb") as f:
                    # noinspection PyTypeChecker
                    pickle.dump({
                        "modelo": modelo,
                        "ids_arquivos": ids_arquivos,
                        "scaler": scaler,
                        "encoder": encoder
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"Modelo treinado e salvo com sucesso em '{modelo_saida}'.")

            except Exception as e:
                print(f"Erro ao salvar o modelo: {e}")

    except Exception as e:
        print(f"Erro durante o treinamento do modelo: {e}")

# Executa o programa
if __name__ == "__main__":
    treinar_modelo_com_db()
