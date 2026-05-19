import os
from src.data.collect import download_data
from src.data.preprocess import load_data, scale_data, create_sequences, train_test_split
from src.model.train import build_model, train_model
from src.model.predict import predict
from src.model.evaluate import evaluate, mape
import matplotlib.pyplot as plt
import joblib
import numpy as np
import json

# Função para inverter a escala de forma dinâmica (melhoria de segurança)
def inverse_transform_single(data, scaler, feature_index):
    """
    Inverte a transformação de apenas uma feature específica.
    """
    dummy = np.zeros((len(data), scaler.n_features_in_))
    dummy[:, feature_index] = data.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index]

TICKER = "PETR4.SA"

# 1. Coletar dados (Certifique-se que download_data aceita o ticker ou edite o collect.py)
print(f"🚀 Iniciando coleta de dados para: {TICKER}")
download_data() 

# 2. Carregar
df = load_data()

# 3. Escalar
scaled, scaler = scale_data(df)

# 4. Criar sequências
X, y = create_sequences(scaled)

# 5. Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 6. Modelo
model = build_model((X_train.shape[1], X_train.shape[2]))

# 7. Treinar
print(f"🧠 Treinando modelo para {TICKER}...")
model = train_model(model, X_train, y_train, validation_split=0.1)

# 8. Prever
pred = predict(model, X_test)

# 9. Inverter a escala ANTES de avaliar (Correção OBRIGATÓRIA)
feature_idx = df.columns.get_loc("Close")
pred_real = inverse_transform_single(pred, scaler, feature_idx)
y_test_real = inverse_transform_single(y_test, scaler, feature_idx)

# 10. Avaliar com valores REAIS
mae, rmse = evaluate(y_test_real, pred_real)
error_mape = mape(y_test_real, pred_real)

print("MAE:", mae)
print("RMSE:", rmse)
print(f"MAPE: {error_mape:.2f}%")

# 11. Salvar métricas reais para consulta na API/Dashboard
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(error_mape)
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)

# 12. Salvar modelo
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/modelo_lstm.keras')

joblib.dump(scaler, 'models/scaler.pkl')

plt.figure(figsize=(10,5))
plt.plot(y_test_real, label='Real')
plt.plot(pred_real, label='Previsto')
plt.legend()
plt.title('Previsão de Preço de Ação')
plt.show()