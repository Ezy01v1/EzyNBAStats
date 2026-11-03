"""Module 10 page: fantasy-point projections with intervals. Educational, never betting advice."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.config import PROCESSED_DIR
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.gamelog import load_player_games
from sportslab.data.schedule import load_schedule
from sportslab.models.fantasy_model import coverage
from sportslab.models.fantasy_projection import project_next_games

CURRENT = "2026-27"
FINISHED = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TRAIN = ["2022-23", "2023-24", "2024-25"]
BLUE, ORANGE, MUTED = "#4C78A8", "#F58518", "#B8BEC6"
TEST_FILE = PROCESSED_DIR / "fantasy_test_predictions.parquet"

st.set_page_config(page_title="Proyecciones fantasy", page_icon="🏀", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Cargando partidos por jugador y calendario…")
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = NbaClient()
    finished = load_player_games(client, FINISHED)
    current = load_player_games(client, [CURRENT], max_age=timedelta(hours=6))
    schedule = load_schedule(client, CURRENT, max_age=timedelta(hours=6))
    return pd.concat([finished, current], ignore_index=True), schedule


@st.cache_data(show_spinner="Ajustando el modelo y calculando proyecciones…")
def get_projections(games: pd.DataFrame, schedule: pd.DataFrame, as_of: str) -> pd.DataFrame:
    # `as_of` is in the signature so the cache refreshes when new games are played
    return project_next_games(games, schedule, TRAIN, FINISHED[-1])


st.title(f"Proyecciones fantasy del próximo partido ({CURRENT})")
st.caption(
    "Uso educativo para entender cuánto varía un jugador de un partido a otro. **No es consejo de "
    "apuestas**: no se compara con líneas ni se recomienda a nadie. Un partido es muy ruidoso: "
    "mira el intervalo, no solo el número."
)

try:
    games, schedule = get_data()
except NbaClientError as exc:
    st.error(
        "No pude descargar los datos y no están en la caché. stats.nba.com bloquea muchas IPs; "
        "ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

as_of = games["game_date"].max()
proj = get_projections(games, schedule, str(as_of.date()))
st.sidebar.write(f"Datos hasta el {as_of.date()}. {len(proj)} jugadores proyectados.")
st.sidebar.caption(
    "Se proyectan jugadores con ≥ 15 minutos de promedio reciente y que jugaron en las últimas 3 "
    "semanas. **Si juega**: los datos no traen lesiones, así que un jugador lesionado hoy "
    "aparece igual."
)
if len(proj) == 0:
    st.warning("No hay próximos partidos en el calendario para proyectar.")
    st.stop()

tab_one, tab_all, tab_valid, tab_limits = st.tabs(
    ["Un jugador", "Todos los próximos partidos", "¿Qué tan bueno es?", "Límites"]
)

with tab_one:
    teams = sorted(proj["team"].unique())
    team = st.selectbox("Equipo", teams)
    options = proj[proj["team"] == team].sort_values("projection", ascending=False)
    name = st.selectbox("Jugador", options["player"].tolist())
    row = options[options["player"] == name].iloc[0]
    venue = "en casa contra" if row["home"] else "de visita contra"
    st.subheader(
        f"{name}: {row['projection']:.1f} puntos fantasy {venue} {row['opponent']} el "
        f"{row['game_date'].date()} (rango del 80 %: {row['low80']:.0f} a {row['high80']:.0f})"
    )
    cols = st.columns(4)
    cols[0].metric("Proyección", f"{row['projection']:.1f}")
    cols[1].metric("Rango 50 %", f"{row['low50']:.0f} - {row['high50']:.0f}")
    cols[2].metric("Rango 80 %", f"{row['low80']:.0f} - {row['high80']:.0f}")
    cols[3].metric("Partidos esta temporada", int(row["n_prior_season"]))
    if row["n_prior_season"] < 10:
        st.warning(
            "Menos de 10 partidos esta temporada: el modelo se apoya mucho en el promedio de la "
            "temporada pasada, así que si el jugador cambió de rol o de equipo, la proyección "
            "puede estar desfasada."
        )
    recent = games[games["player_id"] == row["player_id"]].tail(15)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(d.date()) for d in recent["game_date"]],
            y=recent["fantasy"],
            marker_color=MUTED,
            name="últimos partidos",
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    nxt_label = f"próximo ({row['game_date'].date()})"
    fig.add_trace(
        go.Scatter(
            x=[nxt_label],
            y=[row["projection"]],
            mode="markers",
            marker={"color": BLUE, "size": 11},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [row["high80"] - row["projection"]],
                "arrayminus": [row["projection"] - row["low80"]],
            },
            name="proyección (rango 80 %)",
        )
    )
    fig.update_layout(
        title=f"{name} promedió {recent['fantasy'].mean():.1f} en sus últimos {len(recent)} "
        f"partidos; su rango del 80 % para el próximo mide {row['high80'] - row['low80']:.0f} "
        "puntos de ancho",
        yaxis_title="puntos fantasy",
        height=420,
        legend={"orientation": "h", "y": -0.2},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Puntos fantasy = PTS + 1,2 REB + 1,5 AST + 3 ROB + 3 TAP − pérdidas. El rango sale de los "
        "errores reales del modelo en la temporada pasada, no de una campana supuesta."
    )

with tab_all:
    show = proj.sort_values("projection", ascending=False).head(60).copy()
    show["fecha"] = show["game_date"].dt.date
    show["sede"] = np.where(show["home"], "local", "visita")
    st.dataframe(
        show[
            [
                "player",
                "team",
                "opponent",
                "sede",
                "fecha",
                "projection",
                "low50",
                "high50",
                "low80",
                "high80",
            ]
        ].rename(
            columns={
                "player": "jugador",
                "team": "equipo",
                "opponent": "rival",
                "projection": "proyección",
                "low50": "50 % bajo",
                "high50": "50 % alto",
                "low80": "80 % bajo",
                "high80": "80 % alto",
            }
        ),
        width="stretch",
        height=560,
        column_config={
            c: st.column_config.NumberColumn(format="%.1f")
            for c in ["proyección", "50 % bajo", "50 % alto", "80 % bajo", "80 % alto"]
        },
    )
    st.caption(
        "Los 60 con mayor proyección. 'fecha' es el próximo partido de su equipo en el calendario."
    )

with tab_valid:
    if not TEST_FILE.exists():
        st.info(
            "Corre `python -m uv run python scripts/fantasy_experiment.py` para generar la "
            "validación."
        )
    else:
        test = pd.read_parquet(TEST_FILE)
        rows = []
        for key, label in (
            ("season_avg", "Promedio de la temporada"),
            ("last5", "Últimos 5 partidos"),
            ("ewm10", "Promedio con peso decreciente (~10 partidos)"),
            ("shrunk", "Temporada con contracción"),
            ("ridge", "Modelo (regresión)"),
        ):
            err = test["target"] - test[key]
            rows.append(
                {
                    "método": label,
                    "error absoluto medio": err.abs().mean(),
                    "RMSE": float(np.sqrt((err**2).mean())),
                }
            )
        table = pd.DataFrame(rows)
        c50 = coverage(test["target"], test["low50"], test["high50"])
        c80 = coverage(test["target"], test["low80"], test["high80"])
        width80 = float((test["high80"] - test["low80"]).mean())
        best = table.loc[table["error absoluto medio"].idxmin()]
        by_label = table.set_index("método")["error absoluto medio"]
        last5_note = (
            "los últimos 5 partidos, solos, predicen peor que el promedio de la temporada"
            if by_label["Últimos 5 partidos"] > by_label["Promedio de la temporada"]
            else "los últimos 5 partidos predicen mejor que el promedio de la temporada"
        )
        st.subheader(
            f"En {test['season'].iloc[0]} (prueba final), el mejor método se equivoca "
            f"{best['error absoluto medio']:.1f} puntos fantasy por partido; {last5_note}"
        )
        st.dataframe(table.set_index("método"), width="stretch")
        cols = st.columns(3)
        cols[0].metric("Rango 50 % acierta", f"{c50:.0%}")
        cols[1].metric("Rango 80 % acierta", f"{c80:.0%}")
        cols[2].metric("Ancho medio del 80 %", f"{width80:.0f} pts")
        st.caption(
            f"{len(test):,} partidos de prueba. Un buen intervalo acierta tanto como promete "
            "(50 % → ~50 %, 80 % → ~80 %); un jugador que promedia 25 puede hacer 12 o 38 sin que "
            "nada raro haya pasado."
        )

with tab_limits:
    st.markdown(
        """
**"Si juega."** Los datos solo traen partidos jugados: no se sabe quién está lesionado, descansando
o fuera de la rotación. Un jugador que hoy no juega aparece proyectado igual.

**Sin información de lesiones, alineaciones ni sistemas.** El modelo ve su historial, minutos
recientes, local o visita, descanso y qué tanto permite el rival. Un cambio de rol (por ejemplo,
porque se lesionó un compañero) no lo ve.

**Pocos partidos.** A principios de temporada se apoya en la temporada pasada. Un traspaso o un
cambio de rol lo desfasa.

**Un partido es ruido.** El error típico es de ~8 puntos fantasy sobre un promedio de ~25; la mayor
parte de lo que pasa en un partido no se puede predecir.

**No es consejo de apuestas.** Aquí no hay líneas de casas de apuestas ni recomendaciones.
"""
    )
