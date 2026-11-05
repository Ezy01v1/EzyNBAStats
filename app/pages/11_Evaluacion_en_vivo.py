"""Live evaluation: predictions frozen in Neon before each game, scored against the results."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.models.live_logistic import LOGISTIC_MODEL
from sportslab.service.context import TUNED_ELO
from sportslab.storage import predictions as store

BLUE, ORANGE, GREY = "#4C78A8", "#F58518", "#9E9E9E"
ELO = store.elo_label(TUNED_ELO)
NAMES = {ELO: "Elo", LOGISTIC_MODEL: "Logística (forma + descanso)"}

st.set_page_config(page_title="Evaluación en vivo", page_icon="🏀", layout="wide")
st.title("Evaluación en vivo")
st.caption(
    "Cada mañana se congelan las predicciones en la base de datos, antes de los partidos; "
    "después se comparan con los resultados. Es la prueba honesta: ningún modelo vio el futuro. "
    "Con pocos partidos los números se mueven mucho, por eso cada uno va con su intervalo del 95%."
)


@st.cache_data(ttl=600, show_spinner="Leyendo las predicciones registradas…")
def get_log() -> tuple[pd.DataFrame, pd.DataFrame]:
    with store.connect() as conn:
        return store.read_game_predictions(conn), store.read_fantasy_predictions(conn)


try:
    games, fantasy = get_log()
except Exception as exc:  # no .env, network down, database asleep...
    st.error(f"No pude leer la base de datos de predicciones: {exc}")
    st.stop()

days = games["recorded_on"].nunique() if len(games) else 0
a, b, c = st.columns(3)
a.metric("Días registrados", days)
a.caption(f"desde {games['recorded_on'].min()}" if days else "")
b.metric("Partidos con resultado", int(games.dropna(subset=["home_pts"])["game_id"].nunique()))
c.metric("Juegos de jugador con resultado", int(fantasy["actual"].notna().sum()))

# --- games ----------------------------------------------------------------------------------
st.header("Partidos: ¿quién gana?")
results = store.evaluate_games_by_model(games) if len(games) else {}
if not any(r["n"] for r in results.values()):
    st.info("Todavía no hay partidos con resultado.")
else:
    rows = []
    for model, r in results.items():
        if not r["n"]:
            continue
        lo, hi = r["favorite_won_95"]
        rows.append(
            {
                "Modelo": NAMES.get(model, model),
                "Partidos": r["n"],
                "Brier (moneda = 0.250)": round(r["brier"], 4),
                "Log loss (moneda = 0.693)": round(r["log_loss"], 4),
                "Ganó el favorito": f"{r['favorite_won']:.1%} ({lo:.0%}–{hi:.0%})",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Brier y log loss: más bajo es mejor.")

    cmp = store.compare_models(games, LOGISTIC_MODEL, ELO)
    if cmp["n"]:
        lo, hi = cmp["gain_95"]
        verdict = (
            "la logística es mejor"
            if lo > 0
            else "el Elo es mejor"
            if hi < 0
            else "todavía no se puede distinguir (el intervalo incluye el 0)"
        )
        st.subheader("Logística contra Elo, en los mismos partidos")
        st.markdown(
            f"Ganancia de log loss por partido: **{cmp['gain']:+.4f}** "
            f"(95%: {lo:+.4f} a {hi:+.4f}) en {cmp['n']} partidos → {verdict}. "
            "En la repetición de 2025-26 fue +0.0041 (−0.0016 a +0.0097)."
        )

    st.subheader("Calibración: cuando dice 70%, ¿gana el 70%?")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": GREY}, name="Ideal"
        )
    )
    for (model, r), color in zip(results.items(), (BLUE, ORANGE), strict=False):
        if not r["n"]:
            continue
        cal = r["calibration"]
        fig.add_trace(
            go.Scatter(
                x=cal["mean_pred"],
                y=cal["observed"],
                mode="markers+lines",
                marker={"size": 6 + cal["n"] ** 0.5, "color": color},
                line={"color": color},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": cal["ci_high"] - cal["observed"],
                    "arrayminus": cal["observed"] - cal["ci_low"],
                },
                name=NAMES.get(model, model),
                text=cal["n"],
                hovertemplate="dijo %{x:.0%}, pasó %{y:.0%} (%{text} partidos)<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis={"title": "Probabilidad predicha (local)", "tickformat": ".0%", "range": [0, 1]},
        yaxis={"title": "Frecuencia real", "tickformat": ".0%", "range": [0, 1]},
        height=420,
        margin={"t": 10},
    )
    st.plotly_chart(fig, width="stretch")

# --- fantasy --------------------------------------------------------------------------------
st.header("Fantasy: ¿los intervalos cubren lo que dicen?")
f = store.evaluate_fantasy(fantasy) if len(fantasy) else {"n": 0}
if not f["n"]:
    st.info("Todavía no hay juegos de jugador con resultado.")
else:
    a, b, c = st.columns(3)
    a.metric("Error medio (MAE)", f"{f['mae']:.2f} pts")
    a.caption(f"sesgo {f['bias']:+.2f} (positivo = proyectamos de menos)")
    for col, level in ((b, 50), (c, 80)):
        lo, hi = f[f"coverage{level}_95"]
        col.metric(f"Intervalo del {level}%", f"{f[f'coverage{level}']:.1%}")
        col.caption(f"95%: {lo:.1%}–{hi:.1%}; debería estar cerca de {level}%")
    st.caption(
        f"{f['n']} juegos de {f['players']} jugadores. Las proyecciones son «si juega»: "
        "un jugador que no jugó no cuenta."
    )
