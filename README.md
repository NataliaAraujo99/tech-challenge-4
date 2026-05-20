# 📈 Tech Challenge LSTM - Previsão de Ações com IA

Sistema completo de previsão de preços de ações usando **LSTM (Long Short-Term Memory)** com integração de API FastAPI, dashboard Streamlit e monitoramento em tempo real com Prometheus.

## 🎯 Visão Geral

Este projeto implementa uma solução end-to-end para previsão de preços de ações, abrangendo:
- 📊 **Coleta de dados** via Yahoo Finance
- 🧠 **Modelo LSTM** otimizado para séries temporais
- 📈 **API FastAPI** com documentação Swagger
- 🎨 **Dashboard Streamlit** interativo
- 📡 **Monitoramento com Prometheus** e métricas em tempo real

---

## 📋 Arquitetura do Projeto

```
tech-challenge-4/
├── api/                          # API FastAPI
│   ├── app.py                    # Aplicação principal com endpoints
│   ├── schemas.py                # Modelos de dados Pydantic
│   └── model_loader.py           # Carregamento do modelo LSTM
├── src/
│   ├── data/                     # Processamento de dados
│   │   ├── collect.py            # Download de dados do Yahoo Finance
│   │   ├── preprocess.py         # Normalização e sequências
│   │   └── schemas.py            # Schemas de entrada
│   └── model/                    # Treinamento e avaliação
│       ├── train.py              # Construção e treino do LSTM
│       ├── predict.py            # Inferência
│       └── evaluate.py           # Métricas (MAE, RMSE, MAPE)
├── data/
│   └── raw/                      # Dados brutos do Yahoo Finance
├── models/                       # Modelos treinados
│   ├── modelo_lstm.keras         # Arquivos do modelo
│   ├── scaler.pkl                # Normalizador (StandardScaler)
│   └── metrics.json              # Métricas de treino
├── notebooks/                    # Análise exploratória
├── prometheus/                   # Dados coletados pelo Prometheus
├── logs/
│   └── api.log                   # Logs da API
├── app_streamlit.py              # Dashboard interativo
├── main.py                       # Pipeline completo de treino
├── prometheus.yml                # Configuração do Prometheus
├── Dockerfile                    # Containerização
├── requirements.txt              # Dependências Python
└── README.md                     # Este arquivo
```

---

## 🔄 Pipeline Completo: Do Zero até a Produção

### 1️⃣ **Coleta de Dados**

Os dados são baixados automaticamente do **Yahoo Finance** para o ativo selecionado (padrão: `PETR4.SA`).

```python
# src/data/collect.py
download_data(symbol='PETR4.SA', start='2018-01-01', end=None)
```

**Dados coletados:**
- **Open**: Preço de abertura
- **High**: Preço máximo do dia
- **Low**: Preço mínimo do dia
- **Close**: Preço de fechamento
- **Volume**: Volume de negociação

📁 **Saída:** `data/raw/data.csv`

---

### 2️⃣ **Pré-processamento dos Dados**

Os dados brutos são normalizados e estruturados em sequências para o modelo LSTM.

#### Normalização (`StandardScaler`)
```python
# src/data/preprocess.py
scaled, scaler = scale_data(df)
```

- **StandardScaler**: Normaliza os dados para média=0 e desvio=1
- Essencial para convergência do LSTM
- Salvo em `models/scaler.pkl` para reproduzibilidade

#### Criação de Sequências
```python
X, y = create_sequences(scaled, window=60)
```

- **Window = 60 dias**: O modelo usa os últimos 60 dias para prever o dia seguinte
- **X**: Sequências de 60 timesteps × 5 features
- **y**: Rótulos (preço do dia 61)
- **Proporção treino/teste**: 80/20

---

### 3️⃣ **Arquitetura do Modelo LSTM**

O modelo neural é composto por 2 camadas LSTM com dropout para regularização:

```
Input (60, 5)
    ↓
LSTM (100 unidades, return_sequences=True)
    ↓
Dropout (20%)
    ↓
LSTM (50 unidades)
    ↓
Dropout (20%)
    ↓
Dense (25, ReLU)
    ↓
Dense (1) → Output
```

**Configuração de Treino:**
- **Optimizer**: Adam
- **Loss Function**: Mean Absolute Error (MAE)
- **Epochs**: 100 (com Early Stopping)
- **Batch Size**: 32
- **Validação**: 10% dos dados de treino
- **Callbacks**: Early Stopping + ReduceLROnPlateau

📊 **Saída:** `models/modelo_lstm.keras`

---

### 4️⃣ **Avaliação do Modelo**

Após o treino, o modelo é avaliado no conjunto de teste com as seguintes métricas:

| Métrica | Descrição | Valor |
|---------|-----------|-------|
| **MAE** | Erro Absoluto Médio | em R$ |
| **RMSE** | Raiz do Erro Quadrático Médio | em R$ |
| **MAPE** | Erro Percentual Absoluto Médio | em % |

```python
# main.py
mae, rmse = evaluate(y_test_real, pred_real)
error_mape = mape(y_test_real, pred_real)
```

💾 **Saída:** `models/metrics.json`

---

## 🚀 Como Usar

### Pré-requisitos
- Python 3.12+
- Git
- pip ou conda

### Instalação

1. **Clone o repositório**
```bash
git clone <repositorio>
cd tech-challenge-4
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

---

## 📊 Pipeline de Treino

Execute o pipeline completo para treinar um novo modelo:

```bash
python main.py
```

**O que acontece:**
1. ✅ Baixa dados do Yahoo Finance (`PETR4.SA`)
2. ✅ Normaliza os dados
3. ✅ Cria sequências de 60 timesteps
4. ✅ Divide em treino (80%) e teste (20%)
5. ✅ Constrói e treina o modelo LSTM
6. ✅ Avalia e salva as métricas
7. ✅ Gera gráfico de comparação Real × Previsto
8. ✅ Salva o modelo em `models/`

⏱️ **Tempo esperado:** 5-15 minutos (GPU) ou 30+ minutos (CPU)

---

## 🌐 API FastAPI

A API expõe o modelo treinado via HTTP para consumo em aplicações.

### Iniciar a API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 📡 Endpoints

#### 1. **GET** `/`
Status da API
```bash
curl http://localhost:8000/
```
**Response:**
```json
{"message": "API LSTM rodando"}
```

---

#### 2. **GET** `/health`
Health check para monitoramento
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{"status": "ok"}
```

---

#### 3. **POST** `/predict` ⭐
Realiza predição do preço de fechamento

**Request Body:**
```json
{
  "data": [
    [0.45, 0.47, 0.43, 0.46, 0.50],
    [0.46, 0.48, 0.44, 0.47, 0.55],
    ...
    [0.97, 0.96, 0.95, 0.96, 0.05]
  ]
}
```

**Descrição dos dados:**
- 60 linhas = 60 timesteps (dias)
- 5 colunas = [Open, High, Low, Close, Volume]
- Valores normalizados entre 0 e 1

**Response:**
```json
{
  "prediction": 0.5234,
  "latency_s": 0.0342,
  "cpu_percent": 2.5,
  "ram_mb": 128.34,
  "total_requests": 42
}
```

---

#### 4. **GET** `/metrics`
Métricas do Prometheus (formato texto)
```bash
curl http://localhost:8000/metrics
```
**Output:**
```
# HELP predict_requests_total Total de requisições para o endpoint /predict
# TYPE predict_requests_total counter
predict_requests_total 42.0

# HELP predict_response_time_seconds Tempo de resposta do endpoint /predict em segundos
# TYPE predict_response_time_seconds histogram
predict_response_time_seconds_bucket{le="0.001"} 0.0
...
```

---

### 🧪 Testando os Endpoints

#### Opção 1: Script Python (Recomendado)
```bash
python test_api.py
```
Testa todos os endpoints com dados realistas.

#### Opção 2: cURL
```bash
# Health
curl -X GET "http://localhost:8000/health"

# Predição
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.5, 0.52, ...]]}'
```

#### Opção 3: Swagger UI
1. Abra [http://localhost:8000/docs](http://localhost:8000/docs)
2. Clique em **POST /predict** → **"Try it out"**
3. Cole o JSON de exemplo
4. Clique em **"Execute"**

---

## 🎨 Dashboard Streamlit

Interface gráfica para visualizar previsões em tempo real.

### Iniciar o Dashboard

```bash
streamlit run app_streamlit.py
```

Acesse: [http://localhost:8501](http://localhost:8501)

---

### 📍 Funcionalidades

| Feature | Descrição |
|---------|-----------|
| **Entrada de Ticker** | Busca dados reais do Yahoo Finance |
| **Gráfico de Série Temporal** | Visualiza os últimos 60 dias |
| **Botão "Fazer Previsão"** | Dispara a requisição à API |
| **Resultado Previsto** | Exibe o preço previsto em R$ |
| **Análise de Tendência** | Indica se há alta/baixa/estabilidade |
| **Métricas do Modelo** | MAPE, MAE, confiança do modelo |
| **Monitoramento em Tempo Real** | Latência, CPU, RAM, requisições |

---

### 🖥️ Interface

```
📈 IA de Previsão de Ações

[Digite o Ticker] [🔍 Buscar Dados Oficiais]

📊 Série temporal
[Gráfico do Close Price (últimos 60 dias)]

Dados analisados até: 19/05/2026. A previsão refere-se ao próximo dia útil.

[🔮 Fazer previsão]

Resultado da Predição
┌─────────────────────────────────┬─────────────────────────────────┐
│ 💰 Preço previsto              │ 📈 Tendência de ALTA            │
│ R$ 25.3456                      │ +R$ 0.4521 em relação ao último │
└─────────────────────────────────┴─────────────────────────────────┘

🖥️ Monitoramento do Modelo (Métricas em Tempo Real)
┌─────────────┬──────────────┬────────────┬──────────────┐
│ Latência    │ Requisições  │ CPU        │ RAM          │
│ 0.034s      │ 42           │ 2.5%       │ 128.34 MB    │
└─────────────┴──────────────┴────────────┴──────────────┘
```

---

## 📊 Monitoramento com Prometheus

O Prometheus coleta métricas de performance da API em tempo real.

### Iniciar Prometheus

```bash
./prometheus/prometheus.exe --config.file=prometheus.yml
```

Acesse: [http://localhost:9090](http://localhost:9090)

---

### 🔍 Métricas Coletadas

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `predict_requests_total` | Counter | Total de requisições ao `/predict` |
| `predict_response_time_seconds` | Histogram | Latência em segundos (com buckets) |
| `predict_errors_total` | Counter | Erros ocorridos |
| `predict_processing_active` | Gauge | Requisições ativas simultâneas |
| `process_cpu_usage_percent` | Gauge | % de CPU do processo API |
| `process_memory_usage_bytes` | Gauge | Memória RAM em bytes |

---

### 📈 Queries Úteis no Prometheus

```promql
# Taxa de requisições por segundo
rate(predict_requests_total[5m])

# Latência P95 (95º percentil)
histogram_quantile(0.95, predict_response_time_seconds)

# Erro rate
rate(predict_errors_total[5m]) / rate(predict_requests_total[5m])

# Uso de CPU atual
process_cpu_usage_percent

# Uso de memória em MB
process_memory_usage_bytes / 1024 / 1024
```

---

### 📡 Configuração (prometheus.yml)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi-app'
    static_configs:
      - targets: ['localhost:8000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Funciona com:**
- API FastAPI (`/metrics` endpoint)
- Próprio Prometheus (auto-monitoramento)
- Interval: 15 segundos

---

## 🐳 Docker (Opcional)

### Build
```bash
docker build -t lstm-api .
```

### Run
```bash
docker run -p 8000:8000 lstm-api
```

---

## 📝 Arquivos Importantes

| Arquivo | Propósito |
|---------|-----------|
| `main.py` | Pipeline completo de treino |
| `api/app.py` | API FastAPI com endpoints |
| `app_streamlit.py` | Dashboard interativo |
| `src/data/collect.py` | Download de dados |
| `src/data/preprocess.py` | Normalização e sequências |
| `src/model/train.py` | Construção do LSTM |
| `src/model/evaluate.py` | Métricas de avaliação |
| `prometheus.yml` | Config do Prometheus |
| `test_api.py` | Testes automatizados |

---

## 🔧 Configuração

### Variáveis de Ambiente

Crie um arquivo `.env`:

```env
TICKER=PETR4.SA
API_HOST=localhost
API_PORT=8000
PROMETHEUS_URL=http://localhost:9090
```

### Arquivo `.env.example`

```env
TICKER=PETR4.SA
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 📚 Stack Tecnológico

| Componente | Tecnologia | Versão |
|-----------|-----------|--------|
| **Deep Learning** | TensorFlow/Keras | 2.21.0 / 3.14.0 |
| **Dados** | Pandas, NumPy | 2.2.3, 2.4.4 |
| **API** | FastAPI, Uvicorn | 0.115.0, 0.34.0 |
| **Dashboard** | Streamlit | 1.45.0 |
| **Monitoramento** | Prometheus | 3.11.3 |
| **ML Utils** | scikit-learn | 1.8.0 |
| **Dados Financeiros** | yfinance | 1.3.0 |
| **Python** | 3.12 | - |

---

## 🎯 Workflow Típico

### Ciclo de Desenvolvimento

```
1. Modificar dados
   └─ Editar: src/data/collect.py

2. Treinar modelo
   └─ Executar: python main.py

3. Testar API
   └─ Executar: python test_api.py

4. Visualizar Dashboard
   └─ Abrir: streamlit run app_streamlit.py

5. Monitorar Performance
   └─ Acessar: http://localhost:9090
```

---

## 📊 Exemplo Completo: Previsão Passo a Passo

### 1. Treinar
```bash
python main.py
# Output:
# MAE: 0.5234
# RMSE: 0.8921
# MAPE: 2.34%
```

### 2. Iniciar API
```bash
uvicorn api.app:app --reload
# Uvicorn running on http://127.0.0.1:8000
```

### 3. Testar
```bash
python test_api.py
# ✅ health: PASSOU
# ✅ home: PASSOU
# ✅ predict: PASSOU
# ✅ metrics: PASSOU
```

### 4. Abrir Dashboard
```bash
streamlit run app_streamlit.py
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### 5. Fazer Previsão
- Insira ticker: `PETR4.SA`
- Clique em "🔍 Buscar Dados Oficiais"
- Clique em "🔮 Fazer previsão"
- Veja o resultado em tempo real!

---

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| Modelo não encontrado | Execute `python main.py` primeiro |
| API não conecta | Verifique se `uvicorn` está rodando na porta 8000 |
| Dashboard não abre | Verifique se Streamlit está instalado: `pip install streamlit` |
| Prometheus vazio | Verifique a URL em `prometheus.yml`: `localhost:8000` |
| Dados insuficientes | O Streamlit requer 60 dias de dados. Tente outro ticker |
| MAPE muito alto | Pode indicar dados fora de padrão. Retreine o modelo |

---

## 📖 Referências

- [TensorFlow/Keras LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)
- [Prometheus](https://prometheus.io/docs/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 📄 Licença

MIT

---

## 👥 Autores

**Tech Challenge FIAP** - Equipe de IA Financeira

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `logs/api.log`
2. Consulte o Swagger em `http://localhost:8000/docs`
3. Revise as métricas em `http://localhost:9090`

---

**Última atualização:** 19/05/2026