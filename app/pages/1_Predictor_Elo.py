"""Module 1 page: Elo win-probability baseline. UI only; the logic lives in sportslab."""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import load_games, season_labels
from sportslab.models.elo import expected_home_win, final_ratings
from sportslab.models.experiment import EloExperiment, home_rate_by_season, run_elo_experiment

VALIDATION = ["2023-24", "2024-25"]
TEST = "2025-26"
CURRENT = "2026-27"

st.set_page_config(page_title="Predictor Elo", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="Cargando partidos desde la caché local…")
def get_finished_games() -> pd.DataFrame:
    return load_games(NbaClient(), season_labels(2015, 2025))


@st.cache_data(ttl=3600, show_spinner="Buscando partidos recientes de la temporada en curso…")
def get_current_games() -> pd.DataFrame:
    return load_games(NbaClient(), [CURRENT], max_age=timedelta(hours=6))


@st.cache_data(show_spinner="Ajustando y evaluando el Elo…")
def get_experiment(games: pd.DataFrame) -> EloExperiment:
    return run_elo_experiment(games, validation=VALIDATION, test=TEST)


st.title("Predictor de resultados: Elo")
st.caption(
    "Uso educativo: probabilidades de un modelo simple, no consejo de apuestas. "
    "Un Elo solo ve resultados pasados (no lesiones ni descansos)."
)

try:
    finished = get_finished_games()
except NbaClientError as exc:
    st.error(
        "No pude descargar los partidos y no están en la caché. stats.nba.com bloquea muchas IPs "
        "y rachas de peticiones; ejecuta esto desde tu PC. Alternativa: un CSV de partidos."
    )
    st.code(str(exc))
    st.stop()

exp = get_experiment(finished)
tuned = exp.tuned

current = pd.DataFrame()
try:
    current = get_current_games()
except NbaClientError:
    st.sidebar.warning(f"No pude actualizar la temporada {CURRENT}; uso datos hasta {TEST}.")
all_games = pd.concat([finished, current], ignore_index=True) if not current.empty else finished

st.sidebar.header("Datos")
st.sidebar.write(f"{len(finished):,} partidos de 2015-16 a {TEST}.")
if not current.empty:
    st.sidebar.write(f"+ {len(current)} partidos de {CURRENT} ya jugados.")
st.sidebar.header("Parámetros elegidos (solo validación)")
st.sidebar.write(
    f"K = {tuned.k:.0f} · ventaja de local = {tuned.home_advantage:.0f} · "
    f"regresión a la media = {tuned.carryover}"
)
st.sidebar.caption(
    f"Calentamiento 2015-16 a 2022-23, ajuste con {VALIDATION[0]} y {VALIDATION[1]}, "
    f"prueba final en {TEST}."
)

tab_result, tab_ratings, tab_games = st.tabs(
    ["Resultado del experimento", "Ratings y enfrentamiento", f"Partidos {TEST}"]
)

with tab_result:
    st.subheader(
        f"En {TEST}, el Elo baja el log loss bajo la moneda; 'gana el local' casi no ayuda"
    )
    scores = exp.scores
    cols = st.columns(len(scores))
    for col, (name, row) in zip(cols, scores.iterrows(), strict=True):
        col.metric(name, f"{row['log_loss']:.4f}", help="log loss: menor es mejor")
    st.dataframe(
        scores.rename(columns={"log_loss": "log loss", "brier": "Brier", "accuracy": "acierto"}),
        width="stretch",
    )
    st.caption(
        f"{len(exp.test_predictions)} partidos de prueba (sin sede neutral). "
        "La moneda saca ln 2 = 0.693. El acierto es secundario: no mide qué tan seguro estabas."
    )

    left, right = st.columns(2)
    with left:
        rate = home_rate_by_season(finished) * 100
        fig = go.Figure(go.Bar(x=rate.index, y=rate.values, marker_color="#4C78A8"))
        fig.add_hline(y=50, line_dash="dash")
        fig.update_layout(
            title="La ventaja de local bajó desde 2019-20",
            yaxis={"title": "% victorias del local", "range": [48, 62]},
            height=380,
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        cal = exp.calibration
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="perfecta")
        )
        fig.add_trace(
            go.Scatter(
                x=cal["mean_pred"],
                y=cal["observed"],
                mode="markers",
                marker={"size": cal["n"] / 6 + 6, "color": "#4C78A8"},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": cal["ci_high"] - cal["observed"],
                    "arrayminus": cal["observed"] - cal["ci_low"],
                },
                customdata=cal["n"],
                hovertemplate=(
                    "predicho %{x:.2f}<br>observado %{y:.2f}<br>n=%{customdata}<extra></extra>"
                ),
                name="grupos (tamaño = nº de partidos)",
            )
        )
        fig.update_layout(
            title="Calibración: lo predicho coincide con lo observado (barras = IC 95 %)",
            xaxis_title="probabilidad predicha (local gana)",
            yaxis_title="frecuencia observada",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Los grupos extremos tienen pocos partidos (barras largas): un desvío ahí es ruido, "
        "no una conclusión."
    )

ratings = final_ratings(all_games, tuned)
names = (
    pd.concat(
        [
            all_games[["home_team_id", "home_team"]].set_axis(["team_id", "team"], axis=1),
            all_games[["away_team_id", "away_team"]].set_axis(["team_id", "team"], axis=1),
        ]
    )
    .drop_duplicates("team_id", keep="last")
    .set_index("team_id")["team"]
)
table = (
    pd.DataFrame({"team": names, "rating": pd.Series(ratings)})
    .dropna()
    .sort_values("rating", ascending=False)
    .reset_index(drop=True)
)
table.index = table.index + 1
as_of = all_games["game_date"].max().date()

with tab_ratings:
    st.subheader(f"Ratings Elo al {as_of}")
    st.caption(
        "Al inicio de una temporada los ratings se acercan a 1500 (regresión a la media); "
        "con pocos partidos jugados reflejan mucho el año pasado."
    )
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(table.round({"rating": 0}), width="stretch", height=560)
    with right:
        st.markdown("#### Calculadora de enfrentamiento")
        teams = list(table["team"])
        home = st.selectbox("Local", teams, index=0)
        away = st.selectbox("Visitante", teams, index=min(1, len(teams) - 1))
        neutral = st.checkbox("Sede neutral (sin ventaja de local)")
        if home == away:
            st.info("Elige dos equipos distintos.")
        else:
            r = dict(zip(table["team"], table["rating"], strict=True))
            p = expected_home_win(r[home], r[away], 0.0 if neutral else tuned.home_advantage)
            st.metric(f"Probabilidad de que gane {home}", f"{p:.1%}")
            st.progress(float(p), text=f"{home} {p:.1%} · {away} {1 - p:.1%}")
            st.caption(
                "Modelo simple basado en resultados: no sabe de lesiones, descansos ni "
                "alineaciones. Uso educativo."
            )

with tab_games:
    st.subheader(f"Probabilidades del Elo ajustado en los partidos de {TEST}")
    pred = exp.test_predictions.copy()
    team_filter = st.multiselect(
        "Filtrar por equipo", sorted(set(pred["home_team"]) | set(pred["away_team"]))
    )
    if team_filter:
        pred = pred[pred["home_team"].isin(team_filter) | pred["away_team"].isin(team_filter)]
    view = pred[["game_date", "home_team", "away_team", "p_elo", "home_win"]].rename(
        columns={"p_elo": "P(local gana)", "home_win": "ganó el local"}
    )
    st.dataframe(
        view.sort_values("game_date", ascending=False),
        width="stretch",
        height=520,
        column_config={"P(local gana)": st.column_config.NumberColumn(format="%.3f")},
    )
    st.caption(f"{len(view)} partidos. Predicciones hechas antes de cada partido, sin look-ahead.")
