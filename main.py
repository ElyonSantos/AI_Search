# Avisos de carregamento e informações antes do programa iniciar
print("Carregando... Pode demorar")

import os # Biblioteca de caminhos do python
print("Outras paradas carregadas!") # Aviso de que outras bibliotecas simples foram carregadas

# Aviso de carregamento de parte demorada
print("Carregando programa de indexação...")

import indexacao # Chama o arquivo que cria o banco de dados
import treinamento # Chama o arquivo que faz o treinamento da IA com os embeddings
import testar_embedding_768DM # Chama o arquivo que faz o teste de embeddings 384DM
import testar_carregamento_modelo # Chama o arquivo que faz o teste de carregamento do modelo
import testar_modelo # Chama o arquivo que faz o teste de modelo


# Faz menu
def faz_menu():
    os.system('cls')  # Limpa a tela
    print("Menu:")
    print("1 - Iniciar indexação")
    print("2 - Treinar modelo")
    print("3 - Testar embeddings 768DM")
    print("4 - Testar carregamento do modelo")
    print("5 - Testar modelo .pkl")
    print("6 - Sobre")
    print("7 - Sair")

    return input("Digite: ")


# Função principal
def main():
    while True:
        resultado = faz_menu()

        if resultado == "1":
            indexacao.main()

            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "2":
            print("Treinando o modelo com o banco de dados...")
            treinamento.treinar_modelo_com_db()
            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "3":
            print("Iniciandos teste...")
            testar_embedding_768DM.verificar_embeddings()
            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "4":
            print("Iniciandos teste...")
            testar_carregamento_modelo.main()
            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "5":
            print("Iniciandos teste...")
            testar_modelo.main()
            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "6":
            # Informações extras
            os.system('cls')
            print("Não indexa audios instrumentais; não indexa informações visuais de videos; não indexa .mov"
                  " e outros tipos de video")
            print("Código feito por Deus, ChatGPT, Qwen 2.5 Max, e tudo que tenho direito,"
                  " alem de uma pequena parte feita por Elyon")
            print("\n1 - Voltar")
            print("2 - Sair")

            opcao = input()
            if opcao == "2":
                print("Saindo...")
                break  # Sai do loop e encerra o programa

        elif resultado == "7":
            print("Saindo...")
            break  # Sai do loop e encerra o programa


if __name__ == "__main__":
    main() # Inicia o programa