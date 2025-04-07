import pickle
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

try:
    # Nome do banco de dados e do modelo salvo
    DB_PATH = r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\database\index_arquivos.db"
    MODEL_PATH = r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\modelo\modelo_chatbot.pkl"

    # Carregar o modelo treinado
    with open(MODEL_PATH, "rb") as f:
        modelo_data = pickle.load(f)

    modelo = modelo_data["modelo"]
    ids_arquivos = modelo_data["ids_arquivos"]
    scaler = modelo_data["scaler"]
    encoder = modelo_data["encoder"]

    # Carregar o modelo de embeddings para consultas
    modelo_embedding = SentenceTransformer('all-mpnet-base-v2')

    # Lista de colunas de embeddings usadas no treinamento
    colunas_embeddings = [
        "embedding_nome", "embedding_caminho", "embedding_tipo",
        "embedding_tamanho", "embedding_data", "embedding_status",
        "embedding_conteudo"
    ]
except Exception as e:
    print(f"Erro: {e}")


def gerar_embedding_pergunta(pergunta):
    try:
        """Gera um vetor de consulta usando todas as colunas de embeddings"""

        # Gerar embeddings para cada campo usado no treinamento
        embeddings_pergunta = {coluna: modelo_embedding.encode(pergunta).reshape(1, -1) for coluna in colunas_embeddings}

        # Criar valores padrão para os atributos numéricos e categóricos
        atributos_numericos_padrao = np.array([[0, 0]])  # Exemplo: tamanho do arquivo = 0, timestamp = 0
        atributos_categoricos_padrao = np.zeros((1, 4))  # Codificação OneHot para "desconhecido"

        # 🔹 Ajustando pesos para diferentes embeddings
        PESO_NOME = 5.0
        PESO_CAMINHO = 1.5
        PESO_TIPO = 3.0
        PESO_TAMANHO = 1.0
        PESO_DATA = 1.0
        PESO_STATUS = 1.0
        PESO_CONTEUDO = 6.0  # Aumentamos ainda mais o peso do conteúdo

        # Aplicar pesos
        embeddings_pergunta["embedding_nome"] *= PESO_NOME
        embeddings_pergunta["embedding_caminho"] *= PESO_CAMINHO
        embeddings_pergunta["embedding_tipo"] *= PESO_TIPO
        embeddings_pergunta["embedding_tamanho"] *= PESO_TAMANHO
        embeddings_pergunta["embedding_data"] *= PESO_DATA
        embeddings_pergunta["embedding_status"] *= PESO_STATUS
        embeddings_pergunta["embedding_conteudo"] *= PESO_CONTEUDO

        # Criar vetor final combinando todas as representações com pesos ajustados
        embedding_pergunta_final = np.hstack([
            atributos_numericos_padrao,
            atributos_categoricos_padrao,
            embeddings_pergunta["embedding_nome"],
            embeddings_pergunta["embedding_caminho"],
            embeddings_pergunta["embedding_tipo"],
            embeddings_pergunta["embedding_tamanho"],
            embeddings_pergunta["embedding_data"],
            embeddings_pergunta["embedding_status"],
            embeddings_pergunta["embedding_conteudo"]
        ])

        return embedding_pergunta_final
    except Exception as e:
        print(f"Erro: {e}")


def buscar_arquivos(pergunta, top_k=10):
    try:
        """Busca arquivos mais relevantes para a pergunta em linguagem natural"""

        # Gerar o vetor de consulta
        embedding_pergunta = gerar_embedding_pergunta(pergunta)

        # Encontrar os arquivos mais similares
        distancias, indices = modelo.kneighbors(embedding_pergunta, n_neighbors=top_k)

        # Conectar ao banco de dados para recuperar nomes dos arquivos
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        resultados = []
        for i, indice in enumerate(indices[0]):
            id = ids_arquivos[indice]
            cursor.execute(
                "SELECT nome, caminho, tipo FROM arquivos WHERE id = ?",
                (id,))
            resultado = cursor.fetchone()

            if resultado:
                nome, caminho, tipo = resultado
                similaridade = 1 - distancias[0][i]
                resultados.append((nome, caminho, tipo, similaridade))

        conn.close()

        # 🔹 Reordenar por maior similaridade
        resultados.sort(key=lambda x: x[3], reverse=True)

        # Exibir os resultados ordenados
        print(f"\nResultados para: {pergunta}")
        for i, (nome, caminho, tipo, similaridade) in enumerate(resultados[:top_k]):
            print(f"{i + 1}. {nome} ({caminho}) - Similaridade: {similaridade:.2f}")
    except Exception as e:
        print(f"Erro: {e}")

def main():
    try:
        """Loop interativo para testar o modelo"""
        print("Chatbot de Arquivos - Pergunte sobre seus arquivos!")
        while True:
            pergunta = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
            if pergunta.lower() == "sair":
                print("Encerrando...")
                break
            buscar_arquivos(pergunta)
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
