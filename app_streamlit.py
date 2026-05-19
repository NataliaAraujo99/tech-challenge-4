import streamlit as st
import requests
import numpy as np
import pandas as pd
import json
import yfinance as yf

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="IA Financeira",
    layout="wide"
)
API_URL = "http://127.0.0.1:8000/predict"

# =========================
# CSS CUSTOMIZADO
# =========================
st.markdown("""
<style>

/* FUNDO */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* TÍTULO */
.title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffcc;
}

/* CARDS */
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 10px;
}

/* MÉTRICAS */
.metric {
    font-size: 20px;
    font-weight: bold;
}

/* BOTÕES */
.stButton>button {
    background-color: #00ffcc;
    color: black;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    width: 100%;
}

.stButton>button:hover {
    background-color: #00ccaa;
    color: black;
}

/* INPUTS */
input {
    background-color: #1c1f26 !important;
    color: white !important;
}

/* LABELS */
label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<p class="title">📈 IA de Previsão de Ações</p>', unsafe_allow_html=True)
st.write("Simulação de previsão com modelo LSTM")

# =========================
# GERAR DADOS
# =========================
def buscar_dados_reais(ticker):
    try:
        # Baixa os dados históricos (4 meses para garantir que teremos 60 dias úteis)
        df_ticker = yf.download(ticker, period="4mo", interval="1d")
        if df_ticker.empty:
            st.error(f"Ticker '{ticker}' não encontrado ou sem dados.")
            return None
        
        # Ajuste para versões novas do yfinance: remove o nível extra de colunas (Ticker)
        if isinstance(df_ticker.columns, pd.MultiIndex):
            df_ticker.columns = df_ticker.columns.get_level_values(0)
        
        # Seleciona apenas as colunas necessárias e os últimos 60 registros
        df_ticker = df_ticker[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
        
        if len(df_ticker) < 60:
            st.warning(f"Dados insuficientes para {ticker}. O modelo precisa de 60 dias (obtidos: {len(df_ticker)}).")
            return None
            
        return df_ticker
    except Exception as e:
        st.error(f"Erro ao conectar com Yahoo Finance: {e}")
        return None

if "df" not in st.session_state or st.session_state.df is None:
    # Ticker padrão (ex: PETR4.SA para Petrobras ou AAPL para Apple)
    st.session_state.df = buscar_dados_reais("PETR4.SA")

# =========================
# BOTÕES
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_input("Digite o Ticker da Ação (Yahoo Finance)", value="PETR4.SA")

with col2:
    st.write("##")
    if st.button("🔍 Buscar Dados Oficiais"):
        resultado = buscar_dados_reais(ticker_input)
        if resultado is not None:
            st.session_state.df = resultado
            st.success(f"Dados de {ticker_input} carregados!")

# =========================
# GRÁFICO
# =========================
st.markdown("### 📊 Série temporal")

if st.session_state.df is None:
    st.warning("Não foi possível carregar dados históricos. Tente outro ticker ou verifique a conexão com o Yahoo Finance.")
    st.stop()

ultima_data = st.session_state.df.index[-1].strftime('%d/%m/%Y')
st.caption(f"Dados analisados até: **{ultima_data}**. A previsão refere-se ao próximo dia útil.")

st.line_chart(st.session_state.df["Close"])

# =========================
# MÉTRICAS DE TREINO
# =========================
try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    st.sidebar.markdown("### 🎯 Precisão do Modelo")
    st.sidebar.write(f"**MAPE:** {metrics['mape']:.2e}%")
    st.sidebar.write(f"**MAE:** R$ {metrics['mae']:.4f}")
    
    precisao = 100 - metrics['mape']
    st.sidebar.progress(min(max(precisao/100, 0.0), 1.0))
    st.sidebar.caption(f"Confiança baseada no histórico: {precisao:.2f}%")
except:
    st.sidebar.warning("Métricas de treino não encontradas. Rode o main.py.")


# =========================
# PREVISÃO
# =========================
st.markdown("---")

if st.button("🔮 Fazer previsão"):

    dados = st.session_state.df.values.tolist()
    
    # 🧪 TESTE RÁPIDO
    last_60_days = st.session_state.df
    print(last_60_days[-5:])
    print("Min:", last_60_days.min())
    print("Max:", last_60_days.max())

    # Debug sugerido: Mostra os últimos valores antes de enviar
    with st.expander("🔍 Inspecionar dados enviados para a IA"):
        st.write("Estes são os valores reais que alimentam o modelo (últimos 5 dias):")
        st.dataframe(st.session_state.df.tail(5))
        st.write(f"**Min no período:** {st.session_state.df.values.min():.2f}")
        st.write(f"**Max no período:** {st.session_state.df.values.max():.2f}")

    with st.spinner("🤖 IA analisando dados..."):

        try:
            response = requests.post(
                API_URL,
                json={"data": dados}
            )

            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                
                st.subheader("🎯 Resultado da Predição")

                # Layout em colunas para o resultado
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                    <div class="card">
                        <p style='margin:0; font-size:14px; color:#aaa;'>💰 Preço previsto (Próximo Fechamento)</p>
                        <h1 style='margin:0; color:#00ffcc;'>R$ {pred:.4f}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    # Lógica de tendência baseada no último fechamento, não na média de todo o período.
                    ultimo_fechamento = float(st.session_state.df["Close"].iloc[-1])
                    delta = pred - ultimo_fechamento

                    if abs(delta) <= 1.0:
                        st.info("➡️ Previsão bem próxima do último fechamento")
                    elif delta > 0:
                        st.success(f"📈 Tendência de ALTA (+R$ {delta:.4f} em relação ao último fechamento)")
                    else:
                        st.error(f"📉 Tendência de BAIXA (−R$ {abs(delta):.4f} em relação ao último fechamento)")
                
                # =========================
                # SEÇÃO DE MONITORAMENTO (NOVO)
                # =========================
                st.markdown("---")
                st.subheader("🖥️ Monitoramento do Modelo (Métricas em Tempo Real)")
                
                m1, m2, m3, m4 = st.columns(4)
                
                # Latência
                m1.metric("Latência", f"{result['latency_s']}s")
                # Throughput (Requisições totais nesta sessão da API)
                m2.metric("Total de Requisições", int(result['total_requests']))
                # CPU do Processo
                m3.metric("Uso de CPU", f"{result['cpu_percent']}%")
                # Memória do Processo
                m4.metric("Memória RAM", f"{result['ram_mb']} MB")

            else:
                st.error("Erro na API")

        except requests.exceptions.ConnectionError:
            st.error("❌ Não foi possível conectar à API. Verifique se o Uvicorn está rodando na porta 8000.")
        except Exception as e:
            st.error(f"Erro ao fazer previsão: {e}")