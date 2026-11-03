"""Module 7 page: lineup net ratings with their uncertainty. UI only; logic lives in sportslab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.analysis.lineups import (
    add_uncertainty,
    margin_sd_per_game,
    paired_error_gain,
    paired_halves,
    persistence,
    prediction_errors,
    shrink,
)
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import load_games, season_labels
from sportslab.data.lineups import load_lineups, load_team_ratings

SEASONS = season_labels(2015, 2025)[::-1]  # newest first; 2025-26 is the last complete season
DEFAULT_K = {2: 1600, 3: 400, 5: 400}  # pairs/triples validated in the "¿Persisten?" tab
MUTED, BLUE, ORANGE = "#B8BEC6", "#4C78A8", "#F58518"
KS = (0, 100, 200, 400, 800, 1600, 3200, 1_000_000)
MIN_POSS_BOTH_HALVES = 300

st.set_page_config(page_title="Lineups", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="Cargando lineups desde la caché local…")
def get_lineups(season: str, size: int) -> pd.DataFrame:
    return load_lineups(NbaClient(), season, size)


@st.cache_data(show_spinner="Cargando ratings de equipo…")
def get_teams(season: str) -> pd.DataFrame:
    return load_team_ratings(NbaClient(), season)


@st.cache_data(show_spinner="Cargando partidos…")
def get_margin_sd(season: str) -> float:
    return margin_sd_per_game(load_games(NbaClient(), [season]))


@st.cache_data(
    show_spinner="Comparando primera y segunda mitad (varias peticiones la primera vez)…"
)
def get_persistence(season: str, size: int) -> pd.DataFrame:
    client = NbaClient()
    dates = sorted(load_games(client, [season])["game_date"].dt.date.unique())
    mid = len(dates) // 2
    first = load_lineups(client, season, size, date_to=dates[mid - 1])
    second = load_lineups(client, season, size, date_from=dates[mid])
    team_first = load_team_ratings(client, season, date_to=dates[mid - 1])
    return paired_halves(first, second, team_first, MIN_POSS_BOTH_HALVES)


st.title("Lineups: qué tanto creer en un net rating")
st.caption(
    "Uso educativo. NBA.com entrega solo los ~2000 lineups con más minutos y el net rating de "
    "un lineup tiene un error de muestreo grande: cada cifra va con su intervalo."
)

season = st.sidebar.selectbox("Temporada", SEASONS, index=0)
size = st.sidebar.radio("Jugadores en el lineup", [5, 3, 2], format_func=lambda n: f"{n} jugadores")
min_minutes = st.sidebar.slider("Minutos mínimos", 50, 400, 100, step=25)
k = st.sidebar.slider(
    "Encogimiento hacia el equipo (posesiones)",
    0,
    3200,
    DEFAULT_K[size],
    step=100,
    help="Solo afecta a la columna 'ajustado'. Validado con parejas (1600) y tríos (400).",
)

try:
    lineups = get_lineups(season, size)
    teams = get_teams(season)
    margin_sd = get_margin_sd(season)
except NbaClientError as exc:
    st.error(
        "No pude descargar los datos y no están en la caché. stats.nba.com bloquea muchas IPs; "
        "ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

team_names = sorted(lineups["team"].unique())
team_filter = st.sidebar.multiselect("Equipo", team_names)
st.sidebar.caption(f"Error por partido (SD del margen): {margin_sd:.1f} puntos.")

table = add_uncertainty(lineups[lineups["minutes"] >= min_minutes], margin_sd)
table = table.merge(teams[["team_id", "net_rating"]].rename(columns={"net_rating": "team_net"}))
table["adjusted"] = shrink(table["net_rating"], table["poss"], table["team_net"], k)
if team_filter:
    table = table[table["team"].isin(team_filter)]

tab_rank, tab_persist, tab_cover, tab_gloss = st.tabs(
    ["Ranking con incertidumbre", "¿Persisten?", "Cobertura", "Glosario"]
)

with tab_rank:
    if table.empty:
        st.warning("Ningún lineup cumple los filtros; baja los minutos mínimos.")
    else:
        excl = int(((table["ci_low"] > 0) | (table["ci_high"] < 0)).sum())
        top = table.nlargest(15, "net_rating").iloc[::-1]
        fig = go.Figure(
            go.Scatter(
                x=top["net_rating"],
                y=top["team"] + " · " + top["players"].str.slice(0, 48),
                mode="markers",
                marker={"color": BLUE, "size": 9},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": top["ci_high"] - top["net_rating"],
                    "arrayminus": top["net_rating"] - top["ci_low"],
                },
                customdata=top[["players", "minutes", "poss", "adjusted"]],
                hovertemplate="%{customdata[0]}<br>%{customdata[1]:.0f} min · "
                "%{customdata[2]} poss<br>net rating %{x:+.1f} · ajustado "
                "%{customdata[3]:+.1f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(
            title=f"De {len(table)} lineups con ≥ {min_minutes} min, solo {excl} tienen un "
            f"intervalo que excluye 0 (error mediano ±{1.96 * table['se'].median():.0f})",
            xaxis_title="net rating (puntos por 100 posesiones); barras = IC 95 %",
            height=140 + 34 * len(top),
            margin={"l": 380},
        )
        st.plotly_chart(fig, width="stretch")
        view = table.sort_values("net_rating", ascending=False)[
            ["team", "players", "minutes", "poss", "net_rating", "adjusted", "ci_low", "ci_high"]
        ]
        st.dataframe(
            view.rename(
                columns={
                    "team": "equipo",
                    "players": "jugadores",
                    "minutes": "min",
                    "poss": "posesiones",
                    "net_rating": "net rating",
                    "adjusted": "ajustado",
                    "ci_low": "IC bajo",
                    "ci_high": "IC alto",
                }
            ),
            width="stretch",
            height=420,
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ["net rating", "ajustado", "IC bajo", "IC alto"]
            },
        )
        st.caption(
            "El 'ajustado' acerca el net rating al del equipo (como si el lineup tuviera "
            f"{k} posesiones más iguales al equipo). Los lineups mejor clasificados son los que "
            "más suerte tuvieron además de ser buenos: el ranking por net rating crudo exagera."
        )

with tab_persist:
    st.markdown(
        "¿El net rating de un lineup en la primera mitad de la temporada predice el de la "
        "segunda? Solo parejas y tríos tienen suficientes casos en ambas mitades (de 5 "
        "jugadores, apenas un puñado superan 300 posesiones en las dos)."
    )
    group = st.radio("Tamaño", [2, 3], horizontal=True, format_func=lambda n: f"{n} jugadores")
    if st.button("Calcular (≈ 10-30 s la primera vez)"):
        try:
            joined = get_persistence(season, group)
        except NbaClientError as exc:
            st.error("No pude descargar los datos de ambas mitades.")
            st.code(str(exc))
            st.stop()
        if len(joined) < 30:
            st.warning(f"Solo {len(joined)} lineups en ambas mitades; no alcanza para concluir.")
            st.stop()
        pers = persistence(joined, n_boot=500)
        errors = prediction_errors(joined, KS, "team")
        best_k = int(errors.idxmin())
        gain_raw = paired_error_gain(joined, 0, best_k, n_boot=500)
        gain_team = paired_error_gain(joined, 1_000_000, best_k, n_boot=500)
        st.subheader(
            f"{len(joined)} lineups de {group} jugadores: correlación entre mitades "
            f"{pers['raw'][0]:+.2f}, pero solo {pers['beyond_team'][0]:+.2f} más allá de su equipo"
        )
        cols = st.columns(3)
        cols[0].metric(
            "Entre mitades (Spearman)",
            f"{pers['raw'][0]:+.2f}",
            help=f"IC 95 %: [{pers['raw'][1]:+.2f}, {pers['raw'][2]:+.2f}]",
        )
        cols[1].metric(
            "Más allá del equipo",
            f"{pers['beyond_team'][0]:+.2f}",
            help=f"IC 95 %: [{pers['beyond_team'][1]:+.2f}, {pers['beyond_team'][2]:+.2f}]",
        )
        cols[2].metric("Mejor k", f"{best_k}", help="pseudo-posesiones hacia el equipo")
        fig = go.Figure(
            go.Scatter(
                x=[str(x) if x < 1_000_000 else "equipo solo" for x in errors.index],
                y=errors.values,
                mode="lines+markers",
                line={"color": BLUE},
            )
        )
        fig.update_layout(
            title=f"Error al predecir la 2.ª mitad: el crudo (k=0) es el peor; el mínimo está en "
            f"k={best_k}",
            xaxis_title="k (posesiones hacia el rating del equipo)",
            yaxis_title="error cuadrático ponderado (menor = mejor)",
            height=380,
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"- Ganancia del mejor k sobre el crudo: **{gain_raw[0]:.1f}** "
            f"[{gain_raw[1]:.1f}, {gain_raw[2]:.1f}]\n"
            f"- Ganancia del mejor k sobre usar solo el rating del equipo: **{gain_team[0]:.1f}** "
            f"[{gain_team[1]:.1f}, {gain_team[2]:.1f}]"
        )
        st.caption(
            "Intervalos por bootstrap sobre lineups. Un intervalo que incluye 0 significa que esa "
            "mejora no se distingue del azar. Los lineups que siguen juntos en ambas mitades "
            "no son una muestra al azar (los entrenadores mantienen los que funcionan), así que "
            "la persistencia aquí probablemente es optimista."
        )

with tab_cover:
    all_5 = get_lineups(season, 5)
    covered = all_5["minutes"].sum() / (30 * 82 * 48)
    st.subheader(
        f"Los 2000 lineups de 5 jugadores que entrega NBA.com cubren ~{covered:.0%} de los "
        "minutos de la liga"
    )
    thresholds = pd.DataFrame(
        {
            "minutos mínimos": [25, 50, 100, 200, 300],
            "lineups": [int((all_5["minutes"] >= m).sum()) for m in [25, 50, 100, 200, 300]],
        }
    )
    st.dataframe(thresholds, width="stretch", hide_index=True)
    st.caption(
        f"El lineup más pequeño devuelto tiene {all_5['minutes'].min():.0f} min: los demás no se "
        "descargan. Todo lo de esta página es sobre lineups muy usados, no sobre 'todos los "
        "lineups'. Los minutos de la liga se estiman como 30 equipos × 82 partidos × 48 min."
    )

with tab_gloss:
    st.markdown(
        """
**Net rating** = puntos anotados menos recibidos por 100 posesiones con ese lineup en cancha.

**Error de muestreo.** Un partido (~100 posesiones) tiene un margen con desviación de ~16 puntos.
Medido sobre *n* posesiones, el error estándar es ~16 × √(100 / n): 5 con 1000 posesiones,
10 con 250.
La banda de 95 % es ±1,96 veces ese error. Se calcula con los márgenes reales de la temporada y
algo exagera el ruido (incluye diferencias reales entre equipos).

**Ajustado.** (net rating × posesiones + rating del equipo × k) / (posesiones + k): el net rating
del lineup se acerca al del equipo tanto más cuanto menos posesiones tiene.

**Más allá del equipo.** Correlación entre mitades del rating del lineup *menos* el de su equipo.
Si fuera 0, todo lo que persiste del lineup sería simplemente el equipo.

**Por qué no hay 5 jugadores en la prueba de persistencia.** Los lineups de 5 cambian: entre
mitades casi no se repiten con muchas posesiones.
"""
    )
