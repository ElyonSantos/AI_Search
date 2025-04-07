import pickle

def main():
    try:
        with open(r"C:\Users\elyon\PycharmProjects\AI_Search\projeto_IA\modelo\modelo_chatbot.pkl", "rb") as f:
            modelo_data = pickle.load(f)

        print("Modelo carregado com sucesso!")
        print("Chaves do modelo:", modelo_data.keys())
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main() # Inicia o programa