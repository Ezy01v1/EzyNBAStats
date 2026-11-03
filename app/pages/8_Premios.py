"""Module 8 page: All-Star / All-NBA prediction and the MVP ranking. UI only.

Reads the results written by ``scripts/awards_experiment.py`` (out-of-sample predictions).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportslab.config import PROCESSED_DIR

BLUE, ORANGE, MUTED = "#4C78A8", "#F58518", "#B8BEC6"
FILES = {
    "pred": PROCESSED_DIR / "awards_predictions.parquet",
    "season": PROCESSED_DIR / "awards_per_season.parquet",
    "pooled": PROCESSED_DIR / "awards_pooled.parquet",
    "mvp": PROCESSED_DIR / "awards_mvp.parquet",
}
PANELS = {
    "all_star": ("All-Star", "estadísticas hasta la mitad del calendario", "all_star"),
    "all_nba": ("All-NBA", "estadísticas de toda la temporada regular", "all_nba"),
}
MODEL_LABELS = {
    "pts_only": "Logística solo con puntos",
    "stats": "Logística con estadísticas",
    "logit_all": "Logística + reputación",
    "xgb_all": "XGBoost + reputación",
}

st.set_page_config(page_title="Premios", page_icon="🏀", layout="wide")
st.title("Premios: ¿quién será All-Star o All-NBA?")
st.caption(
    "Uso educativo. Predicciones fuera de muestra: cada temporada se predice con un modelo "
    "entrenado solo con temporadas anteriores y con lo que se sabía en ese momento."
)

if not all(f.exists() for f in FILES.values()):
    st.info(
        "Faltan los resultados. Corre en orden `python -m uv run python "
        "scripts/download_awards.py` (unos 10 minutos la primera vez) y "
        "`python -m uv run python scripts/awards_experiment.py`."
    )
    st.stop()

pred = pd.read_parquet(FILES["pred"])
per_season = pd.read_parquet(FILES["season"])
pooled = pd.read_parquet(FILES["pooled"])
mvp = pd.read_parquet(FILES["mvp"])

panel = st.sidebar.radio(
    "Premio", list(PANELS), format_func=lambda k: f"{PANELS[k][0]} ({PANELS[k][1]})"
)
label_name, _, label_col = PANELS[panel]
seasons = sorted(pred.loc[pred["panel"] == panel, "test_season"].unique(), reverse=True)
season = st.sidebar.selectbox("Temporada", seasons)
model = st.sidebar.selectbox("Modelo", list(MODEL_LABELS), index=2, format_func=MODEL_LABELS.get)

tab_pred, tab_score, tab_mvp, tab_gloss = st.tabs(["Predicciones", "Cómo le fue", "MVP", "Límites"])

with tab_pred:
    g = pred[(pred["panel"] == panel) & (pred["test_season"] == season)].copy()
    g["p"] = g[f"p_{model}"]
    g = g.sort_values("p", ascending=False).reset_index(drop=True)
    r = int(g[label_col].sum())
    top = g.head(r)
    hits = int(top[label_col].sum())
    base = g.sort_values("pts", ascending=False).head(r)
    base_hits = int(base[label_col].sum())
    shown = g.head(max(r + 6, 20)).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=shown["p"] * 100,
            y=shown["player"],
            orientation="h",
            marker_color=[BLUE if v else MUTED for v in shown[label_col]],
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{season}: de {r} {label_name} reales, el modelo puso {hits} entre sus {r} primeros "
        f"(solo ordenar por puntos: {base_hits})",
        xaxis_title="probabilidad del modelo (%) · azul = lo fue de verdad, gris = no",
        height=180 + 22 * len(shown),
    )
    st.plotly_chart(fig, width="stretch")
    missed = g.iloc[r:][g.iloc[r:][label_col]]
    if len(missed):
        st.markdown(
            f"**{label_name} reales que el modelo no puso entre los {r} primeros:** "
            + ", ".join(
                f"{row.player} (puesto {i + r + 1}, {row.p:.0%})"
                for i, row in enumerate(missed.itertuples())
            )
        )
    surprises = top[~top[label_col]]
    if len(surprises):
        st.markdown(
            "**Entre los primeros del modelo pero que no lo fueron:** "
            + ", ".join(f"{row.player} ({row.p:.0%})" for row in surprises.itertuples())
        )
    st.caption(
        "El All-Star mezcla votos de aficionados, cupos por conferencia y reemplazos por lesión; "
        "el modelo solo ve estadísticas y reputación."
    )

with tab_score:
    ps = per_season[per_season["panel"] == panel].drop(columns="panel")
    cols = [c for c in ps.columns if c not in ("season", "winners")]
    names = {
        "score_rank_pts": "Ranking por puntos",
        "score_rank_pie": "Ranking por PIE",
        **{f"p_{k}": v for k, v in MODEL_LABELS.items()},
    }
    means = ps[cols].mean()
    fig = go.Figure()
    for c in cols:
        fig.add_trace(
            go.Scatter(x=ps["season"], y=ps[c], mode="lines+markers", name=names.get(c, c))
        )
    best = means.idxmax()
    fig.update_layout(
        title=f"Precisión@R por temporada de prueba: la media más alta es {names[best]} "
        f"({means.max():.0%}); ordenar por puntos da {means['score_rank_pts']:.0%}",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        yaxis_title="aciertos entre los R primeros / R",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")
    pl = pooled[pooled["panel"] == panel].drop(columns="panel").set_index("model")
    pl.index = [MODEL_LABELS.get(i, i) for i in pl.index]
    st.dataframe(
        pl.rename(
            columns={
                "log_loss": "log loss",
                "brier": "Brier",
                "gain_vs_reference": "mejora vs. solo puntos",
                "gain_low": "IC bajo",
                "gain_high": "IC alto",
            }
        ),
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in pl.columns},
    )
    st.caption(
        "Precisión@R: de los R ganadores reales, cuántos están entre los R primeros del modelo. "
        "La mejora en log loss es por jugador y candidato, con intervalo del 95 %: un intervalo "
        "que incluye 0 no se distingue del azar."
    )

with tab_mvp:
    cols = [c for c in mvp.columns if c != "season"]
    names = {
        "mid_rank_pie": "PIE a mitad de temporada",
        "mid_rank_pts": "Puntos a mitad de temporada",
        "mid_rank_team_w_pct": "% victorias del equipo (mitad)",
        "end_rank_pie": "PIE al final",
        "end_rank_pts": "Puntos al final",
        "end_rank_team_w_pct": "% victorias del equipo (final)",
        "end_rank_model": "Modelo All-NBA (final)",
    }
    view = mvp.set_index("season").rename(columns=names)
    top3 = (mvp[cols] <= 3).sum().rename(index=names)
    n = int(mvp["end_rank_pie"].notna().sum())
    st.subheader(
        f"El MVP real estuvo entre los 3 primeros por PIE al final de la temporada en "
        f"{int(top3['PIE al final'])} de {n} años; a mitad de temporada, en "
        f"{int(top3['PIE a mitad de temporada'])}"
    )
    st.dataframe(view, width="stretch")
    st.caption(
        "Puesto del MVP real entre los ~80 candidatos (1 = el mejor). NBA.com solo da al ganador, "
        "no los votos: con 11 ganadores no se puede entrenar un modelo de MVP, así que esto es "
        "un ranking y no una predicción de probabilidad."
    )

with tab_gloss:
    st.markdown(
        """
**Fuga de información.** El All-Star se vota a mitad de temporada. Entrenar con estadísticas de
toda la temporada dejaría al modelo "ver el futuro". Por eso el panel de All-Star usa
estadísticas hasta la mitad del calendario (en 2020-21, acortada, un 15 de enero no sirve); el de
All-NBA (se anuncia al final) usa la temporada completa.

**Validación temporal.** Para predecir la temporada T se entrena solo con temporadas anteriores
(ventana creciente desde 2019-20). Nunca se mezclan temporadas.

**Precisión@R.** R = número de ganadores reales de esa temporada. Mide cuántos de ellos están
entre los R primeros del ordenamiento del modelo; no se infla por la inmensa mayoría de "no".

**Reputación.** Cuántas veces el jugador fue All-Star (o All-NBA) en temporadas anteriores.
Captura la fama que los votos de aficionados premian. Se cuenta con el historial completo de
cada jugador, solo con temporadas estrictamente anteriores.

**Candidatos.** Los 80 con más puntos por partido en el corte (con mínimo de partidos). Si un
ganador real no estuviera entre ellos, el modelo no podría verlo.

**No sabe.** Cupos por conferencia, votos reales, lesiones que dejan fuera a alguien, cambios de
formato del juego.
"""
    )
