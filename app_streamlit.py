import streamlit as st
import requests
import numpy as np
import pandas as pd
import json
import yfinance as yf
import altair as alt
import joblib

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
    color: black !important;
}

/* GRÁFICOS - Fundo preto */
[data-testid="stChart"] {
    background-color: black !important;
}

[data-testid="stLineChart"] {
    background-color: black !important;
}

/* Plotly charts background */
.plotly-graph-div {
    background-color: black !important;
}

svg {
    background-color: black !important;
}

/* Canvas charts */
canvas {
    background-color: black !important;
}

/* Sidebar titles - Cinza escuro */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #666666 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<p class="title">📈 IA de Previsão de Ações - PETR4</p>', unsafe_allow_html=True)
st.write("Previsão de fechamento usando modelo LSTM")

# =========================
# GERAR DADOS
# =========================
TICKER = "PETR4.SA"

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
        
        # Remove o último dia se estiver com dados incompletos (Open=0)
        # Isso ocorre quando o mercado ainda está aberto
        if df_ticker['Open'].iloc[-1] == 0:
            df_ticker = df_ticker[:-1]
        
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
    with st.spinner("📊 Carregando dados de PETR4.SA..."):
        st.session_state.df = buscar_dados_reais(TICKER)

# =========================
# GRÁFICO
# =========================
st.markdown("### 📊 Série temporal")

if st.session_state.df is None:
    st.warning("Não foi possível carregar dados históricos. Tente outro ticker ou verifique a conexão com o Yahoo Finance.")
    st.stop()

ultima_data = st.session_state.df.index[-1].strftime('%d/%m/%Y')
st.caption(f"Dados analisados até: **{ultima_data}**. A previsão refere-se ao próximo dia útil.")

# Preparar dados para Altair
df_chart = st.session_state.df[['Close']].reset_index()
df_chart.columns = ['Data', 'Close']

# Criar gráfico com Altair
chart = alt.Chart(df_chart).mark_line(
    color='#00ffcc',
    size=2
).encode(
    x=alt.X('Data:T', title='Data', axis=alt.Axis(labelAngle=45, labelColor='white', titleColor='white')),
    y=alt.Y('Close:Q', title='Preço de Fechamento (R$)', axis=alt.Axis(labelColor='white', titleColor='white')),
    tooltip=['Data:T', 'Close:Q']
).properties(
    width='container',
    height=400,
    background='black'
).configure(
    background='black'
).configure_axis(
    grid=True,
    gridColor='#333333'
)

st.altair_chart(chart, use_container_width=True)

# =========================
# MÉTRICAS DE TREINO
# =========================
try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    st.sidebar.markdown("### 🎯 Precisão do Modelo")
    st.sidebar.write(f"**MAPE:** {metrics['mape']:.2f}%")
    st.sidebar.write(f"**MAE:** R$ {metrics['mae']:.2f}".replace('.', ','))
    st.sidebar.write(f"**RMSE:** R$ {metrics['rmse']:.2f}".replace('.', ','))
    
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
        st.write(f"**Primeiros valores a enviar:** {np.array(dados[:2])}")
        st.write(f"**Últimos valores a enviar:** {np.array(dados[-2:])}")

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
                        <h1 style='margin:0; color:#00ffcc;'>R$ {pred:.2f}</h1>
                    </div>
                    """.replace('.', ','), unsafe_allow_html=True)

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
                
                # Adicionar métricas na sidebar
                st.sidebar.markdown("---")
                st.sidebar.subheader("🖥️ Monitoramento em Tempo Real")
                st.sidebar.metric("⏱️ Latência", f"{result['latency_s']}s")
                st.sidebar.metric("📊 Total de Requisições", int(result['total_requests']))
                st.sidebar.metric("💻 Uso de CPU", f"{result['cpu_percent']}%")
                st.sidebar.metric("🧠 Memória RAM", f"{result['ram_mb']} MB")

            else:
                st.error("Erro na API")

        except requests.exceptions.ConnectionError:
            st.error("❌ Não foi possível conectar à API. Verifique se o Uvicorn está rodando na porta 8000.")
        except Exception as e:
            st.error(f"Erro ao fazer previsão: {e}")