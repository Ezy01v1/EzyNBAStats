import streamlit as st

st.set_page_config(page_title="Sports Analytics Lab", page_icon="🏀", layout="wide")

st.title("🏀 Sports Analytics Lab")
st.write(
    "Laboratorio personal de analítica NBA. Cada módulo es una página en la barra lateral, "
    "sobre una capa de datos común con caché en disco."
)

st.subheader("Módulos")
st.markdown(
    """
| # | Módulo | Estado |
|---|---|---|
| 1 | Predictor de resultados (Elo, logística, XGBoost) | **Elo listo**; faltan los otros dos |
| 2 | Shot charts (zonas, dónde vs. cómo) | **listo** |
| 3 | Comparador de jugadores (radar de percentiles) | **listo** |
| 4 | Clustering de roles (con pruebas de validez) | **listo** |
| 5 | Subvalorados | pendiente |
| 6 | Simulador de temporada y playoffs | pendiente |
| 7 | Análisis de lineups (con incertidumbre y persistencia) | **listo** |
| 8 | Predictor de MVP / All-Star | pendiente |
| 9 | Dashboard diario | pendiente |
| 10 | Proyecciones fantasy (educativas) | pendiente |
"""
)
st.caption(
    "Uso educativo. Nada aquí es consejo de apuestas ni promete rentabilidad. "
    "Datos de NBA.com para estudio personal."
)
