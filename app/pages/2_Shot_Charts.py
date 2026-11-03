"""Module 2 page: shot charts and zone efficiency. UI only; the logic lives in sportslab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.analysis.court import court_lines
from sportslab.analysis.shots import (
    BASIC,
    BASIC_RANGE,
    PRIOR_ATTEMPTS,
    league_zone_table,
    selection_vs_making,
    zone_table,
)
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import season_labels
from sportslab.data.players import load_player_stats
from sportslab.data.shots import load_player_shots

SEASONS = season_labels(2015, 2025)[::-1]  # newest first; 2025-26 is the last complete season
MADE, MISSED, MUTED = "#4C78A8", "#F58518", "#B8BEC6"
FEW_SHOTS = 30  # below this a zone's rate is mostly noise: drawn hollow

st.set_page_config(page_title="Shot charts", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="Cargando jugadores desde la caché local…")
def get_players(season: str) -> pd.DataFrame:
    return load_player_stats(NbaClient(), [season])


@st.cache_data(show_spinner="Descargando tiros del jugador (una petición, luego queda en caché)…")
def get_shots(player_id: int, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_player_shots(NbaClient(), player_id, season)


def label(index_value) -> str:
    return " · ".join(index_value) if isinstance(index_value, tuple) else str(index_value)


def add_court(fig: go.Figure) -> None:
    for x, y in court_lines().values():
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", line={"color": "#6B7280", "width": 1}, hoverinfo="skip"
            )
        )


st.title("Shot charts: dónde tira y qué tan bien convierte")
st.caption(
    "Uso educativo. Cada tiro es un dato, pero una zona con pocos tiros dice poco: por eso cada "
    "zona lleva su intervalo y se señala cuando la muestra es chica."
)

season = st.sidebar.selectbox("Temporada", SEASONS, index=0)
granularity = st.sidebar.radio(
    "Detalle de zonas",
    ["Zonas básicas (6)", "Zona + distancia"],
    help="Con más detalle hay menos tiros por zona y más ruido.",
)
keys = BASIC if granularity.startswith("Zonas") else BASIC_RANGE
prior = st.sidebar.slider(
    "Encogimiento (pseudo-tiros al promedio de la liga)",
    0,
    300,
    int(PRIOR_ATTEMPTS),
    help="Solo afecta a la columna 'ajustado': 0 = porcentaje crudo.",
)

try:
    players = get_players(season)
except NbaClientError as exc:
    st.error("No pude cargar la lista de jugadores y no está en la caché.")
    st.code(str(exc))
    st.stop()

candidates = players[players["minutes"] >= 300].sort_values("pie", ascending=False)
labels = {row.player_id: f"{row.player} ({row.team})" for row in candidates.itertuples(index=False)}
player_id = st.selectbox("Jugador", list(labels), format_func=labels.get)

try:
    shots, league = get_shots(int(player_id), season)
except NbaClientError as exc:
    st.error(
        "No pude descargar los tiros y no están en la caché. stats.nba.com bloquea muchas IPs y "
        "rachas de peticiones; ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

name = labels[player_id].split(" (")[0]
if shots.empty:
    st.warning(f"{name} no tiene tiros registrados en {season}.")
    st.stop()

table = zone_table(shots, league, keys, prior_attempts=prior)
split = selection_vs_making(shots, league, keys)
n_made = int(shots["made"].sum())

tab_map, tab_zones, tab_profile, tab_league, tab_gloss = st.tabs(
    ["Mapa de tiros", "Eficiencia por zona", "Dónde tira y cómo convierte", "La liga", "Glosario"]
)

with tab_map:
    best = table.sort_values("freq", ascending=False).iloc[0]
    fig = go.Figure()
    add_court(fig)
    for made, color, symbol, text in ((0, MISSED, "x", "fallado"), (1, MADE, "circle", "anotado")):
        part = shots[shots["made"] == made]
        fig.add_trace(
            go.Scatter(
                x=part["x"],
                y=part["y"],
                mode="markers",
                marker={"color": color, "symbol": symbol, "size": 7, "opacity": 0.55},
                name=text,
                customdata=part[["game_date", "action_type", "distance"]],
                hovertemplate=(
                    f"{text}<br>%{{customdata[0]|%Y-%m-%d}} · %{{customdata[1]}}"
                    "<br>%{customdata[2]} ft<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"{name} en {season}: {len(shots)} tiros, {n_made / len(shots):.1%} de acierto; "
        f"{best['freq']:.0%} desde '{label(best.name)}'",
        xaxis={"range": [-260, 260], "visible": False},
        yaxis={"range": [-55, 330], "visible": False, "scaleanchor": "x"},
        height=620,
        legend={"orientation": "h"},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")
    far = int((shots["y"] > 330).sum())
    st.caption(
        f"Cada punto es un tiro (coordenadas oficiales de NBA.com). {far} tiros desde más allá "
        "del borde de la vista (casi siempre sobre la bocina)."
    )

with tab_zones:
    shown = table.sort_values("league_pps", ascending=False)
    better = shown[(shown["fga"] > 0) & (shown["pps_diff"] > 0)]
    clear = better[better["excludes_league"]]
    n_zones = int((shown["fga"] > 0).sum())
    fig = go.Figure()
    ylabels = [label(i) for i in shown.index]
    fig.add_trace(
        go.Scatter(
            x=shown["league_pps"],
            y=ylabels,
            mode="markers",
            marker={"symbol": "diamond", "size": 11, "color": MUTED, "line": {"color": "#6B7280"}},
            name="promedio de la liga",
        )
    )
    few = shown["fga"] < FEW_SHOTS
    fig.add_trace(
        go.Scatter(
            x=shown["pps"],
            y=ylabels,
            mode="markers",
            marker={
                "size": 10,
                "color": [MUTED if f else MADE for f in few],
                "symbol": ["circle-open" if f else "circle" for f in few],
                "line": {"width": 2, "color": MADE},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": shown["pps_high"] - shown["pps"],
                "arrayminus": shown["pps"] - shown["pps_low"],
            },
            customdata=shown[["fga", "fg_pct"]],
            hovertemplate="%{y}<br>%{x:.2f} pts/tiro<br>%{customdata[0]} tiros · "
            "%{customdata[1]:.1%}<extra></extra>",
            name=name,
        )
    )
    fig.update_layout(
        title=f"{name} rinde sobre la liga en {len(better)} de {n_zones} zonas,<br>"
        f"pero solo en {len(clear)} el intervalo excluye al promedio",
        xaxis_title="puntos por tiro (sin libres); barras = IC 95 % del jugador",
        yaxis={"autorange": "reversed"},
        height=180 + 60 * len(shown),
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(fig, width="stretch")
    view = shown[["fga", "fg_pct", "ci_low", "ci_high", "shrunk_fg_pct", "league_fg_pct", "pps"]]
    view = view.assign(zona=ylabels, pocos=shown["fga"] < FEW_SHOTS).set_index("zona")
    st.dataframe(
        view.rename(
            columns={
                "fga": "tiros",
                "fg_pct": "% crudo",
                "ci_low": "IC bajo",
                "ci_high": "IC alto",
                "shrunk_fg_pct": "% ajustado",
                "league_fg_pct": "% liga",
                "pps": "pts/tiro",
                "pocos": f"< {FEW_SHOTS} tiros",
            }
        ),
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["% crudo", "IC bajo", "IC alto", "% ajustado", "% liga", "pts/tiro"]
        },
    )
    st.caption(
        "Marcadores huecos: menos de 30 tiros, donde el porcentaje casi no informa. '% ajustado' "
        "acerca ese porcentaje al de la liga según el control de la barra lateral. Mirar "
        f"{len(shown)} zonas a la vez da algún 'significativo' por azar: no lo leas como una "
        "conclusión sobre la zona."
    )

with tab_profile:
    sel, mak = split["selection"], split["making"]
    st.subheader(
        f"{name} gana {sel:+.2f} puntos por tiro por DÓNDE tira y {mak:+.2f} por CÓMO convierte"
        f" (IC 95 % [{split['making_low']:+.2f}, {split['making_high']:+.2f}])"
    )
    cols = st.columns(4)
    cols[0].metric("Liga", f"{split['league_pps']:.3f}", help="puntos por tiro, sin libres")
    cols[1].metric("Su mezcla de tiros, a nivel liga", f"{split['league_pps'] + sel:.3f}")
    cols[2].metric("Real", f"{split['actual_pps']:.3f}")
    cols[3].metric("n tiros", f"{split['n']}")
    fig = go.Figure()
    ylabels = [label(i) for i in table.index]
    fig.add_trace(
        go.Bar(y=ylabels, x=table["freq"] * 100, name=name, orientation="h", marker_color=MADE)
    )
    fig.add_trace(
        go.Bar(
            y=ylabels,
            x=table["league_freq"] * 100,
            name="liga",
            orientation="h",
            marker_color=MUTED,
        )
    )
    fig.update_layout(
        title="Perfil de tiro: porcentaje de sus intentos por zona frente a la liga",
        xaxis_title="% de los intentos",
        yaxis={"autorange": "reversed"},
        barmode="group",
        height=140 + 60 * len(table),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Descomposición: 'dónde' = lo que ganaría su mezcla de zonas si convirtiera como el "
        "promedio de la liga; 'cómo' = lo que añade (o resta) convertir mejor que ese promedio "
        "en las mismas zonas. El 'cómo' es mucho más ruidoso: mira su intervalo. Las zonas "
        "no distinguen tiros abiertos de contestados, ni al tirador del sistema que lo rodea."
    )

with tab_league:
    lg = league_zone_table(league, keys).sort_values("pps", ascending=False)
    lg_labels = [label(i) for i in lg.index]
    top, bottom = lg.iloc[0], lg.iloc[-1]
    fig = go.Figure(
        go.Bar(
            y=lg_labels,
            x=lg["pps"],
            orientation="h",
            marker_color=[MADE if v == 3 else MISSED for v in lg["value"]],
            customdata=lg[["fga", "fg_pct"]],
            hovertemplate="%{y}<br>%{x:.2f} pts/tiro<br>%{customdata[0]:,} tiros · "
            "%{customdata[1]:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"En {season}, el tiro más rentable de la liga es '{label(top.name)}'<br>"
        f"({top['pps']:.2f} pts/tiro) y el menos, '{label(bottom.name)}' ({bottom['pps']:.2f})",
        xaxis_title="puntos por tiro, sin libres (azul = triples, naranja = dobles)",
        yaxis={"autorange": "reversed"},
        height=140 + 55 * len(lg),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{int(lg['fga'].sum()):,} tiros de toda la liga. Es lo que SE TIRÓ, no lo que pasaría "
        "si todos tiraran más desde una zona: los tiros cerca del aro son layups y mates abiertos "
        "y muchos tiros de media distancia salen con el reloj agotándose. Además no cuenta libres "
        "ni las faltas que se cobran cerca del aro, que favorecen más a esa zona."
    )

with tab_gloss:
    st.markdown(
        """
**Puntos por tiro (pts/tiro)** = % de acierto × 2 o 3. No incluye libres ni jugadas de 2+1.
Sirve para comparar zonas con distinto valor: 40 % en un triple (1,20) vale más que 50 % en un
doble (1,00).

**Intervalo de Wilson (IC 95 %)** de un porcentaje. Con 10 tiros, un 50 % puede ser verdad entre
~24 % y ~76 %; con 500, entre ~46 % y ~54 %. Si el porcentaje de la liga cae dentro del intervalo,
la diferencia no se distingue del azar.

**Encogimiento** (columna '% ajustado'): (anotados + k × % liga) / (tiros + k). Con pocos tiros se
acerca al promedio de la liga; con muchos, al porcentaje real del jugador. El k por defecto (100)
sale de predecir la 2.ª mitad de la temporada con la 1.ª en 100 jugadores: el error es mínimo
entre 80 y 320, 40 % menor que con el porcentaje crudo.

**Dónde vs. cómo.** Se separa el rendimiento en la calidad de las zonas desde las que tira y su
puntería respecto al promedio de la liga en esas mismas zonas.

**Zonas.** Área restringida (bajo el aro), pintura fuera del área restringida, media distancia,
esquinas (izquierda/derecha) y triple por encima del arco. Los tiros de campo atrás de la
media cancha se ignoran (~30 por temporada en toda la liga).
"""
    )
