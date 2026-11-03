"""Module 6 page: season and playoff simulator. UI only; the logic lives in sportslab."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.config import PROCESSED_DIR
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import load_games, season_labels
from sportslab.data.schedule import load_conferences, load_schedule
from sportslab.models.elo import EloParams
from sportslab.models.season_sim import VALIDATED_RATING_SD, SimConfig, simulate_season
from sportslab.models.season_state import SeasonState, build_season_state

CURRENT = "2026-27"
TUNED_ELO = EloParams(k=24, home_advantage=40, carryover=0.7)  # tuned in Module 1
BLUE, ORANGE, MUTED = "#4C78A8", "#F58518", "#B8BEC6"
BACKTEST_FILE = PROCESSED_DIR / "season_backtest_grid.parquet"

st.set_page_config(page_title="Simulador de temporada", page_icon="🏀", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Cargando calendario y partidos jugados…")
def get_state() -> SeasonState:
    client = NbaClient()
    history = load_games(client, season_labels(2015, 2025))
    schedule = load_schedule(client, CURRENT, max_age=timedelta(hours=6))
    conferences = load_conferences(client, CURRENT)
    return build_season_state(history, schedule, conferences, CURRENT, TUNED_ELO)


@st.cache_data(show_spinner="Simulando miles de temporadas…")
def get_simulation(_state: SeasonState, as_of: str, n_sims: int, rating_sd: float) -> pd.DataFrame:
    # `as_of` is in the signature so the cache refreshes when new games are played
    sim = simulate_season(
        _state.ratings,
        _state.conferences,
        _state.current_wins,
        _state.remaining,
        _state.extra_games,
        SimConfig(n_sims=n_sims, rating_sd=rating_sd, seed=0),
    )
    return sim.table.join(_state.conferences.set_index("team_id")[["team", "conference"]])


st.title(f"Simulador de temporada y playoffs {CURRENT}")
st.caption(
    "Uso educativo, no consejo de apuestas. Juega el resto de la temporada miles de veces con "
    "probabilidades de Elo, luego el play-in y los playoffs. No sabe de lesiones, descansos ni "
    "traspasos."
)

try:
    state = get_state()
except NbaClientError as exc:
    st.error(
        "No pude descargar el calendario y no está en la caché. stats.nba.com bloquea muchas "
        "IPs; ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

n_sims = st.sidebar.select_slider("Simulaciones", [2000, 5000, 10000, 20000], value=10000)
rating_sd = st.sidebar.slider(
    "Incertidumbre del rating (puntos Elo)",
    0,
    150,
    int(VALIDATED_RATING_SD),
    step=25,
    help="0 = tratar el Elo como verdad (intervalos demasiado estrechos). 75 salió del backtest.",
)
st.sidebar.write(f"{state.games_played} partidos jugados hasta el {state.as_of.date()}.")
st.sidebar.write(f"{len(state.remaining)} partidos en el calendario por jugar.")

per_team = state.games_played * 2 / len(state.conferences)
if per_team < 20:
    st.warning(
        f"Solo ~{per_team:.0f} partidos por equipo jugados: los ratings son casi los del año "
        "pasado (regresados hacia la media) y el simulador lo refleja con rangos muy anchos."
    )
if state.owed_total:
    st.warning(
        f"El calendario de NBA.com lista {state.scheduled_min}-{state.scheduled_max} partidos por "
        f"equipo en vez de 82: los otros {state.owed_total} partidos-equipo se simulan contra un "
        "rival promedio."
    )
for message in state.warnings:
    if "instead of 82" not in message:
        st.warning(message)

table = get_simulation(state, str(state.as_of.date()), n_sims, float(rating_sd))
mc_error = 0.5 / np.sqrt(n_sims)

tab_prob, tab_wins, tab_valid, tab_gloss = st.tabs(
    ["Probabilidades", "Victorias esperadas", "¿Se puede confiar?", "Glosario"]
)

with tab_prob:
    ranked = table.sort_values("p_champion", ascending=False)
    top = ranked.iloc[0]
    top5 = ranked["p_champion"].head(5).sum()
    shown = ranked.head(12).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=shown["p_champion"] * 100,
            y=shown["team"],
            orientation="h",
            marker_color=BLUE,
            error_x={
                "type": "data",
                "array": 1.96
                * np.sqrt(shown["p_champion"] * (1 - shown["p_champion"]) / n_sims)
                * 100,
            },
            hovertemplate="%{y}: %{x:.1f}% de ganar el título<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{top['team']} es el favorito con {top['p_champion']:.0%} de ganar el título; los 5 "
        f"primeros suman {top5:.0%}",
        xaxis_title="probabilidad de ser campeón (%) · barras = error de simulación 95 %",
        height=460,
    )
    st.plotly_chart(fig, width="stretch")
    view = ranked[
        [
            "team",
            "conference",
            "current_wins",
            "exp_wins",
            "p_playoffs",
            "p_playin",
            "p_conf_finals",
            "p_finals",
            "p_champion",
        ]
    ]
    st.dataframe(
        view.rename(
            columns={
                "team": "equipo",
                "conference": "conferencia",
                "current_wins": "victorias hoy",
                "exp_wins": "victorias esperadas",
                "p_playoffs": "P(playoffs)",
                "p_playin": "P(play-in)",
                "p_conf_finals": "P(final de conf.)",
                "p_finals": "P(finales)",
                "p_champion": "P(campeón)",
            }
        ),
        width="stretch",
        height=520,
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["victorias esperadas", "P(playoffs)", "P(play-in)"]
            + ["P(final de conf.)", "P(finales)", "P(campeón)"]
        },
    )
    st.caption(
        f"Error de simulación: como máximo ±{1.96 * mc_error:.1%} (95 %) en cualquier "
        "probabilidad; subir las simulaciones lo reduce pero no mejora el modelo. 'P(playoffs)' "
        "incluye ganar el play-in. El Elo solo usa partidos de temporada regular."
    )

with tab_wins:
    ordered = table.sort_values("exp_wins")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ordered["exp_wins"],
            y=ordered["team"],
            mode="markers",
            marker={"color": BLUE, "size": 8},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": ordered["wins_p90"] - ordered["exp_wins"],
                "arrayminus": ordered["exp_wins"] - ordered["wins_p10"],
            },
            name="esperadas (rango 80 %)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ordered["current_wins"],
            y=ordered["team"],
            mode="markers",
            marker={"color": ORANGE, "symbol": "x", "size": 8},
            name="victorias hoy",
        )
    )
    width = (table["wins_p90"] - table["wins_p10"]).mean()
    fig.update_layout(
        title=f"Rango de victorias finales: en promedio {width:.0f} victorias de ancho (80 %)",
        xaxis_title="victorias al final de la temporada regular",
        height=180 + 22 * len(ordered),
        legend={"orientation": "h", "y": -0.05},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "El rango incluye dos fuentes de azar: el resultado de cada partido y qué tan bueno es "
        "realmente cada equipo (el rating es una estimación)."
    )

with tab_valid:
    st.markdown(
        "Antes de confiar en el simulador lo probé en cinco temporadas ya terminadas "
        "(2021-22 a 2025-26), cortando al 10, 30, 50 y 70 % del calendario y usando solo lo "
        "que se sabía entonces."
    )
    if BACKTEST_FILE.exists():
        from sportslab.models.season_backtest import summarize_backtest

        grid = pd.read_parquet(BACKTEST_FILE)
        summary = summarize_backtest(grid, 4000)
        by_sd = summary.groupby("rating_sd").agg(
            cov50=("coverage_50%", "mean"),
            cov80=("coverage_80%", "mean"),
            cov90=("coverage_90%", "mean"),
        )
        fig = go.Figure()
        for col, nominal, color in (
            ("cov50", 0.5, MUTED),
            ("cov80", 0.8, BLUE),
            ("cov90", 0.9, ORANGE),
        ):
            fig.add_trace(
                go.Scatter(
                    x=by_sd.index,
                    y=by_sd[col],
                    mode="lines+markers",
                    name=f"rango nominal {nominal:.0%}",
                    line={"color": color},
                )
            )
            fig.add_hline(y=nominal, line_dash="dot", line_color=color)
        at_zero = by_sd.loc[0, "cov80"]
        at_val = by_sd.loc[int(VALIDATED_RATING_SD), "cov80"]
        fig.update_layout(
            title=f"Con el Elo como verdad, el rango del 80 % solo acierta {at_zero:.0%}; con "
            f"incertidumbre de {int(VALIDATED_RATING_SD)} puntos, {at_val:.0%}",
            xaxis_title="incertidumbre del rating (puntos Elo)",
            yaxis_title="porcentaje de equipos dentro del rango",
            yaxis={"range": [0, 1]},
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Línea punteada = cobertura ideal. El valor de 75 se eligió dejando una temporada "
            "fuera "
            "(se elige con las otras 4 y se mide en la que queda); las dos temporadas de "
            "validación del Elo de la Fase 1 no son una prueba limpia de los parámetros del Elo."
        )
    else:
        st.info(
            "Corre `python -m uv run python scripts/season_backtest.py` (unos minutos) para "
            "generar la tabla de validación y verla aquí."
        )

with tab_gloss:
    st.markdown(
        """
**Monte Carlo.** Jugar la temporada muchas veces al azar y contar cuántas veces pasa cada cosa. Si
un equipo gana el título en 1 900 de 10 000 corridas, su probabilidad es ~19 %.

**Probabilidad de un partido.** `1 / (1 + 10^(−(R_local + 40 − R_visitante) / 400))` con el Elo de
la Fase 1 (ventaja de local 40).

**Incertidumbre del rating.** El Elo es una estimación. En cada corrida cada equipo recibe una
fuerza real distinta, sacada de una campana centrada en su Elo con esta desviación. Sin esto los
intervalos de victorias eran demasiado estrechos: el rango del 80 % solo contenía al 59 % de los
equipos en el backtest.

**Play-in.** Los puestos 7 a 10 de cada conferencia: 7 vs 8 (el ganador es el 7), 9 vs 10, y el
perdedor del primero contra el ganador del segundo por el 8.

**Playoffs.** Cuadro fijo (1-8, 4-5, 3-6, 2-7), series a 7 partidos con formato 2-2-1-1-1;
ventaja de local para el de mejor récord simulado.

**Simplificaciones.** Ratings congelados durante la simulación, desempates al azar, sin lesiones,
descansos ni alineaciones; los partidos que el calendario de NBA.com no lista se juegan contra un
rival promedio.
"""
    )
