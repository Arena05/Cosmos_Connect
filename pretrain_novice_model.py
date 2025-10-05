# Script para pre-entrenar el modelo novato y guardar los artefactos
from app.ml_service import train_and_persist, DEFAULT_NOVICE_MODEL

if __name__ == "__main__":
    model_name, params = DEFAULT_NOVICE_MODEL
    print(f"Entrenando modelo novato: {model_name} con parámetros {params}")
    metrics = train_and_persist(model_name, params, use_user_data=False, user_id=None)
    print("Modelo novato entrenado y artefactos guardados.")
    print(f"Métricas: {metrics}")
