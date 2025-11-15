import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Turnover RH - IA Preditiva", layout="wide")
st.title("Previsão de Turnover (RH)")

# Caminho absoluto para o modelo dentro da pasta src
model_path = os.path.join(os.path.dirname(__file__), "model_rf.pkl")

# Verifica se o modelo existe
if not os.path.exists(model_path):
    st.error("⚠️ O modelo 'model_rf.pkl' não foi encontrado. Rode o script '03_model_ready_data.py' para treinar e salvar o modelo.")
    st.stop()

# Carrega o modelo
rf = joblib.load(model_path)

uploaded = st.file_uploader("Envie um CSV com os campos do dataset sintético")

def predict(df_input):
    X = df_input.drop(columns=["turnover"], errors="ignore")
    preds_proba = rf.predict_proba(X)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)
    out = df_input.copy()
    out["turnover_prob"] = preds_proba
    out["turnover_pred"] = preds
    return out

if uploaded:
    user_df = pd.read_csv(uploaded)
    st.subheader("Dados enviados")
    st.write(user_df.head())
    result = predict(user_df)
    st.subheader("Resultados")
    st.write(result.head())
    st.metric("Média de probabilidade de turnover", f"{result['turnover_prob'].mean():.2f}")
    st.bar_chart(result["turnover_prob"])
