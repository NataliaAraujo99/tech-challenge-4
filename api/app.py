import time
import logging
import os
import psutil
from fastapi import FastAPI
import anyio
from fastapi.responses import PlainTextResponse
import numpy as np
from api.model_loader import load_model, load_scaler
from api.schemas import InputData
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Garantir que a pasta de logs exista
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configuração de Logs
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Métricas do Prometheus
PREDICT_REQUESTS_TOTAL = Counter(
    "predict_requests_total", "Total de requisições para o endpoint /predict"
)
PREDICT_RESPONSE_TIME_SECONDS = Histogram(
    "predict_response_time_seconds",
    "Tempo de resposta do endpoint /predict em segundos",
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)
PREDICT_ERRORS_TOTAL = Counter(
    "predict_errors_total", "Total de erros ocorridos no processamento de previsões"
)
PREDICT_IN_PROGRESS = Gauge(
    "predict_processing_active", "Número de requisições sendo processadas no momento"
)
CPU_USAGE_GAUGE = Gauge("process_cpu_usage_percent", "Uso de CPU do processo da API em porcentagem")
MEMORY_USAGE_GAUGE = Gauge(
    "process_memory_usage_bytes", "Uso de memória RAM do processo da API em bytes"
)

app = FastAPI()

# Carrega o modelo e o scaler ao iniciar a aplicação
model = load_model()
scaler = load_scaler()


@app.get("/")
def home():
    return {"message": "API LSTM rodando"}

@app.get("/health")
async def health():
    """Endpoint para monitoramento de disponibilidade."""
    return {"status": "ok"}

def get_process_metrics():
    process = psutil.Process(os.getpid())
    cpu = process.cpu_percent(interval=None)
    ram = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return cpu, ram

@app.post("/predict", summary="Predição de Preço (LSTM)", tags=["Predictions"])
async def predict(data: InputData):
    """
    Realiza predição do preço de fechamento (Close) usando modelo LSTM.
    
    **Descrição:**
    - Recebe histórico de dados OHLCV (Open, High, Low, Close, Volume) normalizados
    - Processa os dados com o scaler treinado
    - Executa predição usando rede neural LSTM
    - Retorna o preço predito com métricas de performance (latência, CPU, memória)
    
    **Entrada:**
    - `data`: Lista de listas com valores OHLCV. Ex: [[open, high, low, close, volume], ...]
    
    **Saída:**
    - `prediction`: Preço de fechamento predito
    - `latency_s`: Tempo de resposta em segundos
    - `cpu_percent`: Uso de CPU do processo em %
    - `ram_mb`: Uso de memória RAM em MB
    - `total_requests`: Total de requisições processadas
    """
    start_time = time.time()
    PREDICT_REQUESTS_TOTAL.inc()
    
    # Monitora requisições ativas simultâneas
    with PREDICT_IN_PROGRESS.track_inprogress():
        try:
            input_array = np.array(data.data)
            # Log para inspeção: mostra os últimos 2 dias recebidos
            logging.info(f"Input Shape: {input_array.shape} | Últimos valores:\n{input_array[-2:]}")
            
            input_scaled = scaler.transform(input_array)
            input_scaled = np.expand_dims(input_scaled, axis=0)

            # Executa a predição
            prediction_scaled = await anyio.to_thread.run_sync(
                lambda: model.predict(input_scaled, verbose=0))

            # Coleta de recursos do processo
            cpu_usage, memory_usage = get_process_metrics()
            
            CPU_USAGE_GAUGE.set(cpu_usage)
            MEMORY_USAGE_GAUGE.set(memory_usage)

            # Desnormalização robusta usando o número de colunas do scaler
            n_features = scaler.n_features_in_
            dummy = np.zeros((1, n_features))
            # Assumindo 'Close' no índice 3 (Open=0, High=1, Low=2, Close=3, Volume=4)
            dummy[0, 3] = prediction_scaled[0][0]
            prediction_real = scaler.inverse_transform(dummy)[0, 3]

            response_time = time.time() - start_time
            PREDICT_RESPONSE_TIME_SECONDS.observe(response_time)

            logging.info(f"Predict: {prediction_real:.4f} | ResTime: {response_time:.4f}s | CPU: {cpu_usage}%")

            return {
                "prediction": float(prediction_real),
                "latency_s": round(response_time, 4),
                "cpu_percent": cpu_usage,
                "ram_mb": round(memory_usage, 2),
                "total_requests": PREDICT_REQUESTS_TOTAL._value.get()
            }

        except Exception as e:
            PREDICT_ERRORS_TOTAL.inc()
            logging.error(f"Erro na predição: {str(e)}")
            raise e

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Endpoint para o Prometheus coletar as métricas."""
    return generate_latest().decode("utf-8")