"""Module 5 page: who is paid below / above what the market pays for their production.

Needs a salary CSV that YOU provide (see the schema below); nothing is scraped.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.analysis.value import (
    MIN_MINUTES,
    contribution,
    gap_persistence,
    market_fit_summary,
    score_players,
)
from sportslab.config import DATA_DIR
from sportslab.data import NbaClient, NbaClientError
from sportslab.data.players import load_player_stats
from sportslab.data.salaries import OPTIONAL, REQUIRED, match_salaries, read_salaries

SALARY_FILE = DATA_DIR / "external" / "salaries.csv"
BLUE, ORANGE, MUTED = "#4C78A8", "#F58518", "#B8BEC6"
TEMPLATE = "season,player_name,team,salary_usd\n"

st.set_page_config(page_title="Subvalorados", page_icon="🏀", layout="wide")
st.title("Subvalorados: salario frente a producción")
st.caption(
    "Uso educativo. Muestra quién cobra menos (o más) de lo que el mercado paga por una producción "
    "parecida: es un enunciado sobre el mercado, no sobre lo que vale un jugador."
)


@st.cache_data(show_spinner="Cargando estadísticas de jugadores…")
def get_players(season: str) -> pd.DataFrame:
    return load_player_stats(NbaClient(), [season])


st.markdown(
    f"**Qué necesita:** un CSV propio con columnas obligatorias `{', '.join(REQUIRED)}` y "
    f"opcionales `{', '.join(OPTIONAL)}`. NBA.com no publica salarios y no se extraen de sitios "
    "que lo prohíban. Pon el archivo en `data/external/salaries.csv` o súbelo aquí."
)
st.download_button("Descargar plantilla vacía", TEMPLATE, file_name="salaries_template.csv")
st.code("season,player_name,team,salary_usd\n2025-26,Nombre Apellido,ABC,12345678", language="text")

uploaded = st.file_uploader("Subir salaries.csv (no se guarda en el repositorio)", type="csv")
source = None
if uploaded is not None:
    source = io.BytesIO(uploaded.getvalue())
elif SALARY_FILE.exists():
    source = SALARY_FILE
if source is None:
    st.info("Todavía no hay datos de salarios; sin ellos esta página no puede mostrar nada.")
    st.stop()

try:
    salaries = read_salaries(source)
except ValueError as exc:
    st.error(f"El CSV no cumple el esquema: {exc}")
    st.stop()

seasons = sorted(salaries["season"].unique(), reverse=True)
season = st.sidebar.selectbox("Temporada", seasons)
rookie_control = st.sidebar.checkbox(
    "Controlar por contrato de novato",
    value=True,
    help="Los contratos de novato están fijados por el convenio, no por el mercado. Sin "
    "controlarlos, la lista de 'subvalorados' se llena de novatos.",
)
uses_experience = "years_experience" in salaries and salaries["years_experience"].notna().all()
st.sidebar.caption(
    "Novato = hasta 3 años de experiencia (columna del CSV)."
    if uses_experience
    else "Sin la columna `years_experience`, novato se aproxima por edad ≤ 22 (burdo)."
)

try:
    players = get_players(season)
except NbaClientError as exc:
    st.error("No pude descargar las estadísticas de jugadores y no están en la caché.")
    st.code(str(exc))
    st.stop()

matched, csv_left, players_left = match_salaries(salaries[salaries["season"] == season], players)
pool_left = players_left[players_left["minutes"] >= MIN_MINUTES]
st.sidebar.write(
    f"{len(matched)} jugadores con salario cruzado; {len(csv_left)} filas del CSV sin cruzar; "
    f"{len(pool_left)} jugadores de rotación sin salario."
)
if len(matched[matched["minutes"] >= MIN_MINUTES]) < 40:
    st.warning("Menos de 40 jugadores de rotación con salario: el mercado estimado es muy frágil.")
    st.stop()

scored = score_players(matched, rookie_control=rookie_control)
fit = market_fit_summary(matched, rookie_control=rookie_control)

tab_gap, tab_market, tab_match, tab_gloss = st.tabs(
    ["Quién cobra menos / más", "El mercado", "Cruce de nombres", "Límites"]
)

with tab_gap:
    cheap = scored.nsmallest(15, "ratio")
    fig = go.Figure(
        go.Scatter(
            x=cheap["ratio"].iloc[::-1],
            y=(cheap["player"] + " (" + cheap["team"] + ")").iloc[::-1],
            mode="markers",
            marker={"color": BLUE, "size": 9},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (cheap["ratio_high"] - cheap["ratio"]).iloc[::-1],
                "arrayminus": (cheap["ratio"] - cheap["ratio_low"]).iloc[::-1],
            },
        )
    )
    fig.add_vline(x=1, line_dash="dash")
    clear = int((scored["ratio_high"] < 1).sum())
    fig.update_layout(
        title=f"{cheap['player'].iloc[0]} cobra {cheap['ratio'].iloc[0]:.0%} de lo que paga el "
        f"mercado por su producción; {clear} de {len(scored)} tienen un intervalo bajo 100 %",
        xaxis_title="salario real / salario de mercado (barras = IC 95 % de la línea de mercado)",
        height=140 + 30 * len(cheap),
        margin={"l": 240},
    )
    st.plotly_chart(fig, width="stretch")
    view = scored[
        ["player", "team", "age", "minutes", "pie", "salary_usd", "market_salary", "ratio"]
    ].sort_values("ratio")
    st.dataframe(
        view.rename(
            columns={
                "player": "jugador",
                "team": "equipo",
                "age": "edad",
                "minutes": "minutos",
                "pie": "PIE",
                "salary_usd": "salario",
                "market_salary": "salario de mercado",
                "ratio": "real / mercado",
            }
        ),
        width="stretch",
        height=420,
        column_config={
            "salario": st.column_config.NumberColumn(format="$%d"),
            "salario de mercado": st.column_config.NumberColumn(format="$%d"),
            "real / mercado": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "El intervalo recoge la incertidumbre de la línea de mercado pero NO el ruido del PIE: "
        "una buena temporada puntual parece 'subvalorada'. Mira si se repite en otras temporadas "
        "(pestaña 'El mercado')."
    )

with tab_market:
    x = np.log(contribution(scored))
    y = np.log(scored["salary_usd"])
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"color": MUTED, "size": 7},
            text=scored["player"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    order = np.argsort(x.to_numpy())
    fig.add_trace(
        go.Scatter(
            x=x.iloc[order],
            y=np.log(scored["market_salary"]).iloc[order],
            mode="lines",
            line={"color": ORANGE},
            name="mercado (ajustado)",
        )
    )
    fig.update_layout(
        title=f"El mercado explica {fit['r2']:.0%} de la variación de los salarios (n = "
        f"{int(fit['n'])}); +10 % de producción se paga ~{fit['elasticity'] * 10:.0f} % más",
        xaxis_title="log de producción (PIE × minutos)",
        yaxis_title="log del salario",
        height=460,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    others = [s for s in seasons if s != season]
    if others:
        other = st.selectbox("Comparar con otra temporada", others)
        a, b = (other, season) if other < season else (season, other)
        try:
            first = match_salaries(salaries[salaries["season"] == a], get_players(a))[0]
            second = match_salaries(salaries[salaries["season"] == b], get_players(b))[0]
            rho, low, high, n = gap_persistence(first, second, rookie_control=rookie_control)
            st.markdown(
                f"**¿Siguen subvalorados?** Correlación de la brecha salario-mercado entre {a} y "
                f"{b} (mismos {n} jugadores): **{rho:+.2f}** [{low:+.2f}, {high:+.2f}]. Cerca de 0 "
                "significaría que 'subvalorado' era ruido de una temporada."
            )
        except NbaClientError:
            st.info("No pude cargar la otra temporada.")
    st.caption(
        "El mercado se estima solo con jugadores de rotación (≥ 1000 minutos). Producción = PIE × "
        "minutos; PER y BPM no existen en NBA.com."
    )

with tab_match:
    st.markdown("**Filas del CSV que no encontraron jugador** (revisa ortografía o equipo):")
    st.dataframe(csv_left[["player_name", "team", "salary_usd"]].head(100), width="stretch")
    st.markdown("**Jugadores de rotación sin salario en el CSV:**")
    st.dataframe(pool_left[["player", "team", "minutes"]].head(100), width="stretch")

with tab_gloss:
    st.markdown(
        """
**Producción.** PIE × minutos jugados: la parte de los eventos del partido que le corresponde al
jugador, ponderada por cuánto juega.

**Mercado.** Una regresión del log del salario sobre el log de la producción, la edad y una marca de
contrato de novato. 'Salario de mercado' es lo que esa línea paga a alguien con la misma producción.

**Real / mercado.** Menor que 1 = cobra menos que el mercado para su producción.

**Por qué no es 'valor real'.** Cobrar poco puede deberse a un contrato de novato o ya firmado hace
años (el mercado cambió), a un tope salarial, a que el PIE no mide defensa ni espaciado, o a que fue
un año atípico. Es una lista de preguntas, no de respuestas.

**Qué falta.** Datos de contrato (años restantes, tipo de contrato), lesiones, y más temporadas
para ver si la brecha persiste.
"""
    )
