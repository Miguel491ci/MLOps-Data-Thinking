import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Clasificador de Clientes",
    page_icon="🛍️",
    layout="centered",
)

st.title("Clasificador de Segmentos de Clientes")
st.write(
    "Esta aplicación permite probar el modelo entrenado con el dataset Mall Customers. "
    "Introduce los datos de un cliente y el sistema predice a qué segmento pertenece."
)

with st.form("customer_form"):
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.slider("Edad", min_value=18, max_value=80, value=30)
    annual_income_k = st.slider("Ingreso anual en miles de dólares", min_value=10, max_value=150, value=60)
    spending_score = st.slider("Spending Score", min_value=1, max_value=100, value=50)

    submitted = st.form_submit_button("Predecir segmento")

if submitted:
    payload = {
        "gender": gender,
        "age": age,
        "annual_income_k": annual_income_k,
        "spending_score": spending_score,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            segment = result["customer_segment"]
            st.success(f"Cliente clasificado en el segmento {segment}")
            st.json(result)

            st.info(
                "Recuerda: los segmentos fueron creados con clustering. "
                "Para nombrarlos comercialmente, revisa reports/metrics/segment_summary.csv."
            )

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Asegúrate de que FastAPI esté corriendo en el puerto 8000.")