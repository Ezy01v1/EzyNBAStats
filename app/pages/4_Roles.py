"""Module 4 page: role clusters. UI only; the logic lives in sportslab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import silhouette_score

from sportslab.analysis.roles import (
    cluster_profile,
    cross_season_ari,
    feature_set_agreement,
    fit_kmeans,
    k_diagnostics,
    nearest_players,
    principal_components,
    relabel_by_size,
)
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.games import season_labels
from sportslab.data.players import load_player_stats
from sportslab.data.profiles import load_per100, load_shot_zones
from sportslab.features.roles import FEATURE_SUBSETS, FEATURES, build_role_features, standardize

SEASONS = season_labels(2015, 2025)[::-1]  # newest first; 2025-26 is the last complete season
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]
LABELS = {
    "fga100": "Tiros",
    "fg3a100": "Triples",
    "fta100": "Libres",
    "oreb100": "Reb. of.",
    "dreb100": "Reb. def.",
    "ast100": "Asist.",
    "tov100": "Pérdidas",
    "stl100": "Robos",
    "blk100": "Tapones",
    "pf100": "Faltas",
    "share_ra": "% aro",
    "share_paint": "% pintura",
    "share_mid": "% media dist.",
    "share_corner3": "% triple esquina",
    "usg_pct": "USG%",
    "ts_pct": "TS%",
}

st.set_page_config(page_title="Roles de jugadores", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="Cargando estilo de juego de cada jugador…")
def get_features(season: str, min_minutes: int) -> pd.DataFrame:
    client = NbaClient()
    return build_role_features(
        load_player_stats(client, [season]),
        load_per100(client, season),
        load_shot_zones(client, season),
        min_minutes,
    )


@st.cache_data(show_spinner="Calculando diagnósticos (bootstrap, nulo, otra temporada)…")
def get_diagnostics(z: pd.DataFrame, z_prev: pd.DataFrame | None, ks: tuple) -> pd.DataFrame:
    diag = k_diagnostics(z, ks, n_boot=50, n_null=10)
    if z_prev is not None:
        diag["cross_season_ari"] = [cross_season_ari(z_prev, z, k) for k in ks]
    return diag


def strength(silhouette: float) -> str:
    if silhouette < 0.25:
        return "estructura débil"
    return "estructura razonable" if silhouette < 0.5 else "estructura fuerte"


st.title("Roles de jugadores: ¿hay grupos naturales?")
st.caption(
    "Uso educativo. k-means siempre devuelve k grupos, haya o no grupos reales: por eso cada "
    "agrupación va con pruebas de que no es un artefacto. Agrupa por CÓMO juega, no por qué tan "
    "bueno es."
)

season = st.sidebar.selectbox("Temporada", SEASONS, index=0)
min_minutes = st.sidebar.slider("Minutos mínimos", 800, 2000, 1000, step=100)
k = st.sidebar.slider("Número de grupos (k)", 2, 8, 3)

try:
    feats = get_features(season, min_minutes)
except NbaClientError as exc:
    st.error(
        "No pude descargar los datos y no están en la caché. stats.nba.com bloquea muchas IPs; "
        "ejecuta esto desde tu PC."
    )
    st.code(str(exc))
    st.stop()
if len(feats) < 60:
    st.warning(f"Solo {len(feats)} jugadores cumplen {min_minutes} min; baja el umbral.")
    st.stop()

z, _, _ = standardize(feats)
model = fit_kmeans(z, k)
labels = relabel_by_size(model.labels_)
sil = float(silhouette_score(z, labels))
st.sidebar.write(f"{len(feats)} jugadores · {len(FEATURES)} variables")

tab_map, tab_profile, tab_like, tab_real, tab_gloss = st.tabs(
    ["Mapa de roles", "Perfil de cada grupo", "Jugadores parecidos", "¿Son reales?", "Glosario"]
)

with tab_map:
    coords, explained = principal_components(z)
    fig = go.Figure()
    for c in range(k):
        mask = labels == c
        fig.add_trace(
            go.Scatter(
                x=coords.loc[mask, "PC1"],
                y=coords.loc[mask, "PC2"],
                mode="markers",
                name=f"Grupo {c + 1} ({int(mask.sum())})",
                marker={"color": PALETTE[c % len(PALETTE)], "size": 8, "opacity": 0.8},
                text=feats.loc[mask, "player"] + " (" + feats.loc[mask, "team"] + ")",
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Con k={k} hay {strength(sil)} (silueta {sil:.2f}): los grupos son cortes de un "
        "continuo, no islas",
        xaxis_title=f"Componente 1 ({explained[0]:.0%} de la variación)",
        yaxis_title=f"Componente 2 ({explained[1]:.0%} de la variación)",
        height=560,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Cada punto es un jugador proyectado a 2 dimensiones; las dos componentes resumen "
        f"{explained.sum():.0%} de la información, así que grupos que parecen mezclados aquí "
        "pueden estar separados en las otras dimensiones. Silueta: <0.25 débil, 0.25-0.5 "
        "razonable, >0.5 fuerte."
    )

with tab_profile:
    prof = cluster_profile(z, labels)
    sizes = prof.pop("size")
    names = [f"Grupo {c + 1} ({sizes[c]})" for c in prof.index]
    fig = go.Figure(
        go.Heatmap(
            z=prof.to_numpy(),
            x=[LABELS[c] for c in prof.columns],
            y=names,
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "z"},
            hovertemplate="%{y}<br>%{x}: %{z:+.2f} desv. est.<extra></extra>",
        )
    )
    fig.update_layout(
        title="Perfil de cada grupo: cuántas desviaciones estándar sobre (rojo) o bajo (azul) "
        "el promedio de la liga",
        yaxis={"autorange": "reversed"},
        height=140 + 55 * k,
    )
    st.plotly_chart(fig, width="stretch")
    for c in range(k):
        row = prof.loc[c]
        top = row.reindex(row.abs().sort_values(ascending=False).index)[:3]
        distinct = ", ".join(f"{'+' if v > 0 else '−'}{LABELS[f]}" for f, v in top.items())
        players = feats.loc[labels == c].nlargest(8, "minutes")["player"].tolist()
        st.markdown(f"**Grupo {c + 1}** ({sizes[c]}): se distingue por {distinct}.")
        st.caption("Más minutos: " + ", ".join(players))
    st.caption(
        "Los nombres de los roles (p. ej. 'pívots protectores del aro') los pones tú al leer el "
        "perfil; el algoritmo solo da números de grupo."
    )

with tab_like:
    player = st.selectbox("Jugador", sorted(feats["player"]))
    near = nearest_players(z, feats["player"], player, n=8)
    group_of = dict(zip(feats["player"], labels + 1, strict=True))
    near["grupo"] = near["player"].map(group_of)
    st.markdown(f"**{player}** está en el Grupo {group_of[player]}. Los jugadores más parecidos:")
    st.dataframe(
        near.rename(columns={"player": "jugador", "distance": "distancia"}),
        width="stretch",
        column_config={"distancia": st.column_config.NumberColumn(format="%.2f")},
    )
    st.caption(
        "Distancia euclidiana entre perfiles estandarizados (menor = más parecido). Un jugador "
        "poco común tiene vecinos lejanos: la distancia importa más que el orden."
    )

with tab_real:
    st.markdown(
        "Tres pruebas de que los grupos no son un artefacto: (1) **silueta** frente a datos "
        "revueltos sin estructura, (2) **estabilidad** al remuestrear jugadores y (3) acuerdo "
        "con la **temporada anterior**."
    )
    if st.button("Calcular diagnóstico (≈ 30-60 s)"):
        z_prev = None
        if season != SEASONS[-1]:
            prev = SEASONS[SEASONS.index(season) + 1]
            try:
                z_prev, _, _ = standardize(get_features(prev, min_minutes))
            except NbaClientError:
                st.info(f"No pude cargar {prev}; se omite la comparación entre temporadas.")
        ks = tuple(range(2, 9))
        diag = get_diagnostics(z, z_prev, ks)
        best = diag.loc[diag["ari_mean"].idxmax()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diag["k"], y=diag["silhouette"], name="silueta (datos reales)"))
        fig.add_trace(
            go.Scatter(
                x=diag["k"],
                y=diag["null_silhouette"],
                name="silueta (datos revueltos)",
                line={"dash": "dash"},
            )
        )
        fig.add_trace(go.Scatter(x=diag["k"], y=diag["ari_mean"], name="estabilidad (bootstrap)"))
        if "cross_season_ari" in diag:
            fig.add_trace(
                go.Scatter(x=diag["k"], y=diag["cross_season_ari"], name="acuerdo otra temporada")
            )
        fig.update_layout(
            title=f"La agrupación más estable es k={int(best['k'])} (estabilidad "
            f"{best['ari_mean']:.2f}); con más grupos se vuelve inestable",
            xaxis_title="k",
            yaxis_title="valor (1 = perfecto; silueta y ARI)",
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(diag.round(3), width="stretch")
        agree = feature_set_agreement(z, FEATURE_SUBSETS, k)
        st.markdown(
            f"**¿Cambia el agrupamiento (k={k}) si cambio las variables?** ARI contra el completo:"
        )
        st.dataframe(agree.rename("ARI").round(2).to_frame(), width="stretch")
        st.caption(
            "ARI ~1 = mismos grupos; ~0 = coinciden como el azar. Si un grupo desaparece al quitar "
            "un bloque de variables, depende de esa decisión mía, no de los datos. El nulo solo "
            "demuestra que las variables están correlacionadas entre sí, no que haya islas."
        )

with tab_gloss:
    st.markdown(
        """
**Estandarizar (z-score).** Cada variable se resta su media y se divide por su desviación
estándar. Sin esto, mandarían las variables con unidades grandes (tiros por 100 posesiones) sobre
las pequeñas (TS% ≈ 0,57).

**k-means.** Elige k centros y asigna cada jugador al más cercano, minimizando la distancia total
dentro de los grupos. Siempre devuelve k grupos.

**Silueta** = (b − a) / max(a, b). `a` = distancia media a los de su grupo; `b` = distancia media
al grupo vecino más cercano. Cerca de 1: bien separado; cerca de 0: frontera arbitraria.

**Datos revueltos (nulo).** Cada columna se baraja por separado: mismas distribuciones, cero
relación entre variables. Si la silueta real no supera a la revuelta, no hay estructura.

**ARI (índice de Rand ajustado).** Qué tanto coinciden dos agrupaciones del mismo conjunto, sin
importar cómo se numeren los grupos. 1 = idénticas; ~0 = como el azar.

**Variables.** Por 100 posesiones: tiros, triples, libres, rebotes ofensivos y defensivos,
asistencias, pérdidas, robos, tapones, faltas. Proporción de intentos por zona (aro, pintura,
media distancia, triple de esquina; el triple sobre el arco es el resto). USG% y TS%.
"""
    )
