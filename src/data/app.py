from fastapi import FastAPI
import numpy as np
from api.model_loader import load_model
from api.schemas import InputData

app = FastAPI()

# Carrega o modelo ao iniciar a aplicação
model = load_model()

@app.get("/")
def home():
    return {"message": "API LSTM rodando"}

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.data)
    input_array = np.expand_dims(input_array, axis=0) # Adiciona a dimensão do batch
    prediction = model.predict(input_array)
    return {"prediction": float(prediction[0][0])}