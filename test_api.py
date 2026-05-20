"""
Script para testar os endpoints da API LSTM
Execute com: python test_api.py
"""
import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000"

def test_health():
    """Testa o endpoint de health check"""
    print("\n=== Testando /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_home():
    """Testa o endpoint home"""
    print("\n=== Testando / ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict():
    """Testa o endpoint de predição com dados simulados"""
    print("\n=== Testando /predict ===")
    
    # Gera 60 timesteps com dados simulados normalizados entre 0 e 1
    np.random.seed(42)
    timesteps = 60
    features = 5  # OHLCV
    
    # Simula dados OHLCV normalizados
    data = np.random.uniform(0.4, 0.8, size=(timesteps, features)).tolist()
    
    payload = {"data": data}
    
    print(f"Enviando {timesteps} timesteps com {features} features (OHLCV)...")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response:")
        print(f"  Prediction: {result.get('prediction', 'N/A'):.6f}")
        print(f"  Latency: {result.get('latency_s', 'N/A')}s")
        print(f"  CPU: {result.get('cpu_percent', 'N/A')}%")
        print(f"  RAM: {result.get('ram_mb', 'N/A')} MB")
        print(f"  Total Requests: {result.get('total_requests', 'N/A')}")
    else:
        print(f"Erro: {response.text}")
    
    return response.status_code == 200

def test_metrics():
    """Testa o endpoint de métricas do Prometheus"""
    print("\n=== Testando /metrics ===")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        lines = response.text.split('\n')
        print(f"Primeiras 10 linhas de métricas:")
        for line in lines[:10]:
            if line and not line.startswith('#'):
                print(f"  {line}")
        print(f"  ... (total de {len(lines)} linhas)")
    else:
        print(f"Erro: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 60)
    print("Testando API LSTM")
    print("=" * 60)
    
    try:
        results = {
            "health": test_health(),
            "home": test_home(),
            "predict": test_predict(),
            "metrics": test_metrics(),
        }
        
        print("\n" + "=" * 60)
        print("RESUMO DOS TESTES")
        print("=" * 60)
        for endpoint, passed in results.items():
            status = "✅ PASSOU" if passed else "❌ FALHOU"
            print(f"{endpoint}: {status}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não consegui conectar à API!")
        print(f"Verifique se a API está rodando em {BASE_URL}")
        print("Inicie com: uvicorn api.app:app --reload")
