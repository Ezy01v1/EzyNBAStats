"""Module 9 page: daily dashboard. Descriptive, plus Elo probabilities (educational, no betting)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.analysis.daily import (
    add_probabilities,
    games_on,
    last_game_day,
    next_game_day,
    standings,
    surprises,
    top_performances,
)
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.gamelog import load_player_games
from sportslab.data.schedule import schedule_to_games
from sportslab.models.season_sim import SimConfig, simulate_season
from sportslab.models.season_state import SeasonState
from sportslab.service.context import CURRENT, FRESHNESS, TUNED_ELO, load_context

BLUE, ORANGE = "#4C78A8", "#F58518"

st.set_page_config(page_title="Dashboard diario", page_icon="🏀", layout="wide")


@st.cache_data(ttl=1800, show_spinner="Cargando calendario, resultados y estadísticas…")
def get_data():
    # Same loading as the API (sportslab.service.context), so both show the same numbers
    client = NbaClient()
    ctx = load_context(client, CURRENT)
    players = load_player_games(client, [CURRENT], max_age=FRESHNESS)
    return ctx.schedule, ctx.conferences, players, ctx.state, ctx.pre_game


@st.cache_data(show_spinner="Simulando el resto de la temporada…")
def get_playoff_odds(_state: SeasonState, as_of: str) -> pd.Series:
    table = simulate_season(
        _state.ratings,
        _state.conferences,
        _state.current_wins,
        _state.remaining,
        _state.extra_games,
        SimConfig(n_sims=3000, seed=0),
    ).table
    return table["p_playoffs"]


st.title("Dashboard diario")
st.caption(
    "Uso educativo. Resume lo que pasó y lo que viene; las probabilidades son del Elo de la "
    "Fase 1 y no saben de lesiones, descansos ni alineaciones. No es consejo de apuestas."
)

try:
    schedule, conferences, players, state, pre = get_data()
except NbaClientError as exc:
    st.error(
        "No pude descargar los datos y no están en la caché. stats.nba.com bloquea muchas IPs; "
        "ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

today = pd.Timestamp.today().normalize()
last_day = last_game_day(schedule)
next_day = next_game_day(schedule, today)
names = conferences.set_index("team_id")["team"]
abbr = players.drop_duplicates("team_id").set_index("team_id")["team"]
label = lambda tid: abbr.get(tid, names.get(tid, "TBD"))  # noqa: E731

st.sidebar.write(f"Hoy: {today.date()}")
st.sidebar.write(f"Última jornada con partidos: {last_day.date() if last_day is not None else '-'}")
st.sidebar.write(f"Próxima jornada: {next_day.date() if next_day is not None else '-'}")
if last_day is not None and (today - last_day).days > 2:
    st.warning(
        f"Los datos llegan hasta el {last_day.date()}, hace {(today - last_day).days} días: "
        "puede que aún no se hayan cargado los resultados más recientes."
    )
if state.owed_total:
    st.warning(
        f"El calendario lista {state.scheduled_min}-{state.scheduled_max} partidos por equipo en "
        "vez de 82; las probabilidades de playoffs lo tratan simulando los que faltan contra un "
        "rival promedio."
    )

col_left, col_right = st.columns(2)

with col_left:
    if last_day is not None:
        st.subheader(f"Resultados del {last_day.date()}")
        day = games_on(schedule, last_day)
        day = day[day["played"]].merge(pre[["game_id", "p_home"]], on="game_id", how="left")
        day["local"] = day["home_team_id"].map(label)
        day["visitante"] = day["away_team_id"].map(label)
        day["marcador"] = (
            day["home_pts"].astype(int).astype(str)
            + " - "
            + day["away_pts"].astype(int).astype(str)
        )
        day["favorito ganó"] = ((day["p_home"] >= 0.5) == (day["home_pts"] > day["away_pts"])).map(
            {True: "sí", False: "no"}
        )
        st.dataframe(
            day[["local", "marcador", "visitante", "p_home", "favorito ganó"]].rename(
                columns={"p_home": "P(local) previa"}
            ),
            width="stretch",
            hide_index=True,
            column_config={"P(local) previa": st.column_config.NumberColumn(format="%.2f")},
        )
        sur = surprises(day.assign(played=True), 0.35)
        if len(sur):
            st.caption(
                "Sorpresas (ganó quien tenía ≤ 35 %): "
                + ", ".join(
                    f"{r.local if r.home_pts > r.away_pts else r.visitante} ({r.p_winner:.0%})"
                    for r in sur.itertuples()
                )
            )
        else:
            st.caption("Sin sorpresas (nadie con ≤ 35 % ganó).")

with col_right:
    if next_day is not None:
        st.subheader(f"Partidos del {next_day.date()}")
        upcoming = add_probabilities(
            games_on(schedule, next_day), state.ratings, TUNED_ELO.home_advantage
        )
        upcoming["local"] = upcoming["home_team_id"].map(label)
        upcoming["visitante"] = upcoming["away_team_id"].map(label)
        upcoming["favorito"] = [
            f"{h} ({p:.0%})" if p >= 0.5 else f"{a} ({1 - p:.0%})"
            for h, a, p in zip(
                upcoming["local"], upcoming["visitante"], upcoming["p_home"], strict=True
            )
        ]
        st.dataframe(
            upcoming[["visitante", "local", "p_home", "favorito"]].rename(
                columns={"p_home": "P(local gana)"}
            ),
            width="stretch",
            hide_index=True,
            column_config={"P(local gana)": st.column_config.NumberColumn(format="%.2f")},
        )
        st.caption(
            "Probabilidad del Elo (ventaja de local 40). En el Módulo 1 mejoró el log loss de 0.69 "
            "(moneda) a ~0.60: sabe algo, pero ignora lesiones y descansos."
        )
    else:
        st.info("No hay próximos partidos en el calendario.")

st.subheader("Posiciones por conferencia")
odds = get_playoff_odds(state, str(last_day.date()) if last_day is not None else "")
played = schedule_to_games(schedule, CURRENT)
table = standings(played, conferences)
table["elo"] = table["team_id"].map(state.ratings)
table["p_playoffs"] = table["team_id"].map(odds)
tabs = st.tabs(sorted(table["conference"].unique()))
for tab, conf in zip(tabs, sorted(table["conference"].unique()), strict=True):
    with tab:
        view = table[table["conference"] == conf].reset_index(drop=True)
        view.index = view.index + 1
        st.dataframe(
            view[
                ["team", "wins", "losses", "win_pct", "last_n", "diff", "elo", "p_playoffs"]
            ].rename(
                columns={
                    "team": "equipo",
                    "wins": "G",
                    "losses": "P",
                    "win_pct": "%",
                    "last_n": "últimos 10",
                    "diff": "dif/partido",
                    "elo": "Elo",
                    "p_playoffs": "P(playoffs)",
                }
            ),
            width="stretch",
            height=560,
            column_config={
                "%": st.column_config.NumberColumn(format="%.3f"),
                "dif/partido": st.column_config.NumberColumn(format="%+.1f"),
                "Elo": st.column_config.NumberColumn(format="%.0f"),
                "P(playoffs)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
st.caption(
    "Con pocos partidos jugados la tabla es casi puro azar: P(playoffs) viene de simular el resto "
    "de la temporada (Módulo 6) y se apoya sobre todo en el Elo de la temporada pasada. El "
    "desempate real de la NBA no está implementado."
)

if last_day is not None:
    st.subheader(f"Mejores actuaciones del {last_day.date()} (puntos fantasy)")
    top = top_performances(players, last_day, 10)
    fig = go.Figure(
        go.Bar(
            x=top["fantasy"].iloc[::-1],
            y=(top["player"] + " (" + top["team"] + ")").iloc[::-1],
            orientation="h",
            marker_color=BLUE,
            customdata=top[["pts", "reb", "ast"]].iloc[::-1],
            hovertemplate="%{y}: %{x:.1f}<br>%{customdata[0]} pts · %{customdata[1]} reb · "
            "%{customdata[2]} ast<extra></extra>",
        )
    )
    best = top.iloc[0]
    fig.update_layout(
        title=f"{best['player']} lideró la jornada con {best['fantasy']:.1f} puntos fantasy "
        f"({int(best['pts'])} pts, {int(best['reb'])} reb, {int(best['ast'])} ast)",
        xaxis_title="puntos fantasy",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Lo que no está: lesiones, quién descansa, alineaciones y cuotas de apuestas. Los datos vienen "
    "de NBA.com con un retraso de hasta 3 horas."
)
