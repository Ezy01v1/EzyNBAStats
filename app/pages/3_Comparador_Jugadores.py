"""Module 3 page: player comparator. UI only; the logic lives in sportslab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.analysis.comparator import (
    MIN_MINUTES,
    RADAR_METRICS,
    binned_means,
    correlation_direction,
    eligible_pool,
    percentile_profile,
    rank_correlation,
    top_overlap,
)
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import season_labels
from sportslab.data.players import load_player_stats

SEASONS = season_labels(2015, 2025)[::-1]  # newest first; 2025-26 is the last complete season
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
MUTED = "#B8BEC6"

st.set_page_config(page_title="Comparador de jugadores", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="Cargando estadísticas de jugadores desde la caché local…")
def get_players(season: str) -> pd.DataFrame:
    return load_player_stats(NbaClient(), [season])


@st.cache_data(show_spinner="Calculando intervalos (bootstrap)…")
def get_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    return rank_correlation(x, y)


def agreement(rho: float) -> str:
    size = abs(rho)
    return "coinciden poco" if size < 0.3 else "coinciden en parte" if size < 0.6 else "coinciden"


st.title("Comparador de jugadores")
st.caption(
    "Uso educativo. Los percentiles comparan a cada jugador con los de su misma temporada que "
    "jugaron lo suficiente; no dicen quién es mejor en un contexto concreto."
)

season = st.sidebar.selectbox("Temporada", SEASONS, index=0)
min_minutes = st.sidebar.slider(
    "Minutos mínimos para entrar al grupo de comparación", 300, 2500, MIN_MINUTES, step=100
)
st.sidebar.caption(
    "Con pocos minutos los porcentajes son inestables. 1000 min ≈ 40 partidos de 25 min."
)

try:
    players = get_players(season)
except NbaClientError as exc:
    st.error(
        "No pude descargar las estadísticas y no están en la caché. stats.nba.com bloquea muchas "
        "IPs y rachas de peticiones; ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()

pool = eligible_pool(players, min_minutes)
if len(pool) < 30:
    st.warning(f"Solo {len(pool)} jugadores cumplen {min_minutes} min; baja el umbral.")
    st.stop()
st.sidebar.write(f"{len(pool)} jugadores en el grupo (de {len(players)} con minutos en {season}).")

candidates = players[players["minutes"] >= 300].sort_values("pie", ascending=False)
labels = {row.player_id: f"{row.player} ({row.team})" for row in candidates.itertuples(index=False)}
default_ids = list(pool.nlargest(2, "pie")["player_id"])
picked = st.multiselect(
    "Jugadores (2 a 4)",
    options=list(labels),
    default=default_ids,
    format_func=labels.get,
    max_selections=4,
)

tab_cmp, tab_usage, tab_net, tab_gloss = st.tabs(
    ["Comparación", "¿Más uso, menos eficiencia?", "Net rating vs. PIE", "Glosario"]
)

with tab_cmp:
    if len(picked) < 2:
        st.info("Elige al menos dos jugadores.")
    else:
        chosen = players.set_index("player_id").loc[picked].reset_index()
        pct = percentile_profile(chosen, pool)
        names = [labels[i] for i in picked]
        theta = [RADAR_METRICS[m] for m in RADAR_METRICS]

        if len(picked) == 2:
            ahead = int((pct.iloc[0] > pct.iloc[1]).sum())
            title = (
                f"{chosen['player'][0]} queda por encima de {chosen['player'][1]} "
                f"en {ahead} de {len(RADAR_METRICS)} métricas"
            )
        else:
            title = (
                f"Percentiles frente a {len(pool)} jugadores con ≥ {min_minutes} min en {season}"
            )

        fig = go.Figure()
        for i, name in enumerate(names):
            r = pct.iloc[i].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=[*r, r[0]],
                    theta=[*theta, theta[0]],
                    name=name,
                    fill="toself",
                    opacity=0.45,
                    line={"color": PALETTE[i]},
                    hovertemplate="%{theta}: percentil %{r:.0f}<extra>" + name + "</extra>",
                )
            )
        fig.update_layout(
            title=title,
            polar={"radialaxis": {"range": [0, 100], "ticksuffix": ""}},
            height=520,
        )
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("**Valores reales**")
            raw = chosen[["player", "team", "gp", "minutes", *RADAR_METRICS]].copy()
            raw["minutes"] = raw["minutes"].round(0)
            st.dataframe(
                raw.rename(columns={"player": "jugador", "team": "equipo", "gp": "PJ"})
                .rename(columns={"minutes": "minutos", **RADAR_METRICS})
                .set_index("jugador"),
                width="stretch",
            )
            st.markdown("**Percentiles (0-100)**")
            view = pct.rename(columns=RADAR_METRICS).round(0)
            view.index = chosen["player"]
            st.dataframe(view, width="stretch")
        low_minutes = chosen.loc[chosen["minutes"] < min_minutes, "player"].tolist()
        if low_minutes:
            st.warning(
                f"Con menos de {min_minutes} min: {', '.join(low_minutes)}. Su percentil se "
                "calcula frente al grupo, pero con esa muestra los porcentajes son poco fiables."
            )
        st.caption(
            "El net rating es del EQUIPO por 100 posesiones con el jugador en cancha: depende de "
            "con quién comparte minutos, no es un atributo individual. PIE reemplaza a PER/BPM, "
            "que NBA.com no publica (ver Glosario)."
        )

with tab_usage:
    rho, low, high = get_correlation(pool["usg_pct"], pool["ts_pct"])
    verdict = {
        "none": "sin relación detectable",
        "negative": "más uso, menos eficiencia",
        "positive": "más uso, más eficiencia",
    }[correlation_direction(low, high)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pool["usg_pct"] * 100,
            y=pool["ts_pct"] * 100,
            mode="markers",
            marker={"color": MUTED, "size": 7},
            text=pool["player"],
            hovertemplate="%{text}<br>USG %{x:.1f}% · TS %{y:.1f}%<extra></extra>",
            name="grupo",
        )
    )
    for i, pid in enumerate(picked):
        row = players[players["player_id"] == pid].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row.usg_pct * 100],
                y=[row.ts_pct * 100],
                mode="markers+text",
                text=[row.player],
                textposition="top center",
                marker={"color": PALETTE[i], "size": 11},
                name=row.player,
            )
        )
    fig.update_layout(
        title=f"Uso y eficiencia en {season}: {verdict} (Spearman {rho:+.2f}, "
        f"IC 95 % [{low:+.2f}, {high:+.2f}])",
        xaxis_title="USG% (porción de posesiones usadas)",
        yaxis_title="TS% (eficiencia anotadora)",
        height=480,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    bins = binned_means(pool, "usg_pct", "ts_pct")
    fig = go.Figure(
        go.Scatter(
            x=bins["usg_pct_mean"] * 100,
            y=bins["ts_pct_mean"] * 100,
            mode="lines+markers",
            line={"color": PALETTE[0]},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (bins["ci_high"] - bins["ts_pct_mean"]) * 100,
                "arrayminus": (bins["ts_pct_mean"] - bins["ci_low"]) * 100,
            },
            customdata=bins["n"],
            hovertemplate=(
                "USG medio %{x:.1f}%<br>TS medio %{y:.1f}%<br>n=%{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="TS% medio por quintil de uso (barras = IC 95 % de la media)",
        xaxis_title="USG% medio del quintil",
        yaxis_title="TS% medio",
        height=360,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{len(pool)} jugadores. Una correlación cercana a 0 no dice que usar más posesiones sea "
        "gratis: los jugadores de alto uso suelen ser los mejores anotadores y enfrentan las "
        "mejores defensas, y los pívots de bajo uso anotan cerca del aro. Es una relación entre "
        "jugadores, no el efecto de subirle el uso a un mismo jugador."
    )

with tab_net:
    rho, low, high = get_correlation(pool["net_rating"], pool["pie"])
    n_shared, shared = top_overlap(pool, "net_rating", "pie", n=20)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pool["pie"] * 100,
            y=pool["net_rating"],
            mode="markers",
            marker={"color": MUTED, "size": 7},
            text=pool["player"],
            hovertemplate="%{text}<br>PIE %{x:.1f}<br>net rating %{y:+.1f}<extra></extra>",
        )
    )
    for i, pid in enumerate(picked):
        row = players[players["player_id"] == pid].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row.pie * 100],
                y=[row.net_rating],
                mode="markers+text",
                text=[row.player],
                textposition="top center",
                marker={"color": PALETTE[i], "size": 11},
            )
        )
    fig.update_layout(
        title=f"En {season}, net rating y PIE {agreement(rho)}: Spearman {rho:+.2f} "
        f"[{low:+.2f}, {high:+.2f}], {n_shared}/20 en ambos top-20",
        xaxis_title="PIE (%)",
        yaxis_title="Net rating (puntos por 100 posesiones)",
        height=480,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Jugadores en ambos top-20: {', '.join(sorted(shared)) or 'ninguno'}. El net rating "
        "mezcla al jugador con sus compañeros y rivales; PIE solo mide su propia participación "
        "en eventos del box score. Que difieran no dice cuál tiene razón."
    )

with tab_gloss:
    st.markdown(
        """
**TS%** = PTS / (2 · (FGA + 0,44 · FTA)). Eficiencia anotadora que cuenta triples y libres.

**USG%** = porción de las posesiones de su equipo que el jugador termina (tiro, libres o pérdida)
mientras está en cancha.

**AST% / REB%** = porción de las canastas de sus compañeros que asiste / de los rebotes
disponibles que captura mientras juega.

**Net rating** = (puntos anotados − recibidos por 100 posesiones) del *equipo* con el jugador en
cancha. No es una estadística individual.

**PIE** = porción de los eventos del box score del partido (puntos, rebotes, asistencias, robos,
tapones, pérdidas, tiros fallados...) que le corresponde al jugador. Es el análogo de NBA.com de
PER.

**PER y BPM no están en NBA.com.** PER (Hollinger) es una fórmula por minuto normalizada a 15 de
promedio de liga; BPM es una regresión del box score calibrada con el net rating. Vienen de
Basketball Reference; si quieres usarlos, exporta un CSV de ahí y lo importamos.

**Percentil** = porción del grupo con un valor menor (los empates cuentan por la mitad). Pierde la
magnitud: dos jugadores en el percentil 98 pueden estar muy lejos entre sí en el valor real.
"""
    )
