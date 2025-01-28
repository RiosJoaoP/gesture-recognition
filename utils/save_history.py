import pickle
import os

# Salvar o objeto history em um arquivo pickle
def save_training_history(history, output_dir):
    history_path = os.path.join(output_dir, "training_history.pkl")
    with open(history_path, "wb") as file:
        pickle.dump(history.history, file)
    print(f"Histórico de treinamento salvo em: {history_path}")