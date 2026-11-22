"""
╔══════════════════════════════════════════════════════════════╗
║  SISTEMA DE MANUTENÇÃO PREDITIVA COM IA                      ║
║  Máquinas de Hemodiálise e Respiradores Pulmonares           ║
║  FAESA – Engenharia da Computação III – 2026                 ║
╚══════════════════════════════════════════════════════════════╝

Execute com:
    streamlit run app.py

Datasets de referência:
  - NASA C-MAPSS: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
  - Predictive Maintenance: https://www.kaggle.com/c/predictive-maintenance/data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_generator import (
    gerar_dados_hemodialise,
    gerar_dados_respirador,
    snapshot_atual,
    historico_maquina,
)
from models.ml_models import PipelinePreditivo


# ─── Configuração da Página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="ManuIA – Manutenção Preditiva Hospitalar",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Customizado ──────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Fonte e paleta geral */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Cards de métricas */
  [data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
  }
  [data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem !important;
    font-weight: 600;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Cabeçalho principal */
  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border: 1px solid #1e40af;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
  }
  .main-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
  }
  .main-header p {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0.3rem 0 0;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Badge de status */
  .badge-normal    { background:#052e16; color:#4ade80; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-atencao   { background:#1c1917; color:#fbbf24; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-alerta    { background:#431407; color:#fb923c; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-critico   { background:#450a0a; color:#f87171; padding:2px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

  /* Tabela de equipamentos */
  .equip-table { width:100%; border-collapse: collapse; font-size: 0.88rem; background: #1e293b; border-radius: 10px; overflow: hidden; }
  .equip-table th { color: #e2e8f0 !important; font-weight:600; text-transform:uppercase; font-size:0.72rem;
                    letter-spacing:0.06em; padding: 10px 14px; background: #0f172a; border-bottom: 2px solid #3b82f6; }
  .equip-table td { padding: 10px 14px; border-bottom: 1px solid #334155; color: #f1f5f9 !important; background: #1e293b; }
  .equip-table tr:hover td { background: #334155 !important; color: #ffffff !important; }

  /* Seção de análise */
  .section-title {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569; margin: 1.5rem 0 0.8rem;
  }

  div[data-testid="stTabs"] button { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ─── Cache: Geração de Dados e Treinamento ────────────────────────────────────

@st.cache_data(show_spinner="Gerando dados dos sensores...")
def carregar_dados():
    df_hd = gerar_dados_hemodialise(n_maquinas=10, seed=42)
    df_resp = gerar_dados_respirador(n_maquinas=10, seed=99)
    return df_hd, df_resp


@st.cache_resource(show_spinner="Treinando modelos de IA...")
def treinar_modelos(df_hd, df_resp):
    pipeline_hd = PipelinePreditivo("Hemodiálise")
    pipeline_hd.treinar(df_hd)

    pipeline_resp = PipelinePreditivo("Respirador")
    pipeline_resp.treinar(df_resp)

    return pipeline_hd, pipeline_resp


# ─── Funções Auxiliares ───────────────────────────────────────────────────────

CORES_STATUS = {
    "Normal":  "#4ade80",
    "Atenção": "#fbbf24",
    "Alerta":  "#fb923c",
    "Crítico": "#f87171",
}

BADGE_CLASS = {
    "Normal":  "badge-normal",
    "Atenção": "badge-atencao",
    "Alerta":  "badge-alerta",
    "Crítico": "badge-critico",
}


def badge(status):
    cls = BADGE_CLASS.get(status, "badge-normal")
    return f'<span class="{cls}">{status}</span>'


def gauge_rul(rul_val, max_rul=300, titulo="RUL"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(rul_val),
        title={"text": titulo, "font": {"size": 14, "color": "#94a3b8"}},
        number={"suffix": " ciclos", "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, max_rul], "tickcolor": "#475569",
                     "tickfont": {"size": 10, "color": "#475569"}},
            "bar": {"color": "#3b82f6"},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 20],       "color": "#450a0a"},
                {"range": [20, 50],      "color": "#431407"},
                {"range": [50, 100],     "color": "#1c1917"},
                {"range": [100, max_rul],"color": "#052e16"},
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 2},
                "thickness": 0.75,
                "value": 20,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
    )
    return fig


def grafico_sensor_historico(hist_df, col_sensor, titulo, unidade=""):
    fig = go.Figure()

    # Linha principal do sensor
    fig.add_trace(go.Scatter(
        x=hist_df["ciclo"],
        y=hist_df[col_sensor],
        mode="lines",
        name=titulo,
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
    ))

    # Pontos de anomalia
    anomalias = hist_df[hist_df.get("anomalia", pd.Series(False, index=hist_df.index))]
    if len(anomalias) > 0:
        fig.add_trace(go.Scatter(
            x=anomalias["ciclo"],
            y=anomalias[col_sensor],
            mode="markers",
            name="Anomalia",
            marker=dict(color="#f87171", size=7, symbol="x"),
        ))

    fig.update_layout(
        title=dict(text=f"{titulo} ({unidade})" if unidade else titulo,
                   font=dict(size=13, color="#94a3b8"), x=0),
        height=220,
        margin=dict(t=40, b=30, l=50, r=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(title="Ciclo", gridcolor="#1e293b", color="#475569"),
        yaxis=dict(gridcolor="#1e293b", color="#475569"),
        legend=dict(font=dict(size=10)),
        showlegend=True,
    )
    return fig


def grafico_distribuicao_status(snap_df, titulo):
    contagem = snap_df["urgencia"].value_counts().reset_index()
    contagem.columns = ["Status", "Qtd"]
    ordem = ["Normal", "Atenção", "Alerta", "Crítico"]
    contagem["Status"] = pd.Categorical(contagem["Status"], categories=ordem, ordered=True)
    contagem = contagem.sort_values("Status")

    cores = [CORES_STATUS.get(s, "#94a3b8") for s in contagem["Status"]]

    fig = go.Figure(go.Bar(
        x=contagem["Status"],
        y=contagem["Qtd"],
        marker_color=cores,
        text=contagem["Qtd"],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=13, color="#94a3b8"), x=0),
        height=260,
        margin=dict(t=40, b=30, l=40, r=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b", color="#475569"),
        yaxis=dict(gridcolor="#1e293b", color="#475569"),
        showlegend=False,
    )
    return fig


def grafico_importancia(imp_df):
    fig = go.Figure(go.Bar(
        x=imp_df["importancia_pct"].head(8),
        y=imp_df["sensor"].head(8),
        orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.1f}%" for v in imp_df["importancia_pct"].head(8)],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        title=dict(text="Importância dos Sensores (Random Forest)",
                   font=dict(size=13, color="#94a3b8"), x=0),
        height=280,
        margin=dict(t=40, b=20, l=160, r=60),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#1e293b", color="#475569", title="Importância (%)"),
        yaxis=dict(gridcolor="#1e293b", color="#475569"),
    )
    return fig


def grafico_rul_scatter(res_df):
    fig = go.Figure()
    for urgencia, cor in CORES_STATUS.items():
        subset = res_df[res_df["urgencia"] == urgencia]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset["rul"],
                y=subset["rul_predito"],
                mode="markers",
                name=urgencia,
                marker=dict(color=cor, size=6, opacity=0.7),
            ))

    # Linha de referência y = x (predição perfeita)
    lim = max(res_df["rul"].max(), res_df["rul_predito"].max())
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode="lines",
        name="Referência",
        line=dict(color="#475569", dash="dash", width=1),
    ))

    fig.update_layout(
        title=dict(text="RUL Real vs. RUL Predito pelo Modelo",
                   font=dict(size=13, color="#94a3b8"), x=0),
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(title="RUL Real (ciclos)", gridcolor="#1e293b", color="#475569"),
        yaxis=dict(title="RUL Predito", gridcolor="#1e293b", color="#475569"),
        legend=dict(font=dict(size=10)),
    )
    return fig


def grafico_anomalia_score(hist_df, titulo):
    if "score_anomalia" not in hist_df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["ciclo"],
        y=hist_df["score_anomalia"],
        mode="lines",
        name="Score",
        line=dict(color="#a855f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(168,85,247,0.08)",
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#fb923c",
                  annotation_text="Limiar Alerta", annotation_font_color="#fb923c")
    fig.add_hline(y=0.85, line_dash="dash", line_color="#f87171",
                  annotation_text="Limiar Crítico", annotation_font_color="#f87171")

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=13, color="#94a3b8"), x=0),
        height=220,
        margin=dict(t=40, b=30, l=50, r=80),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(title="Ciclo", gridcolor="#1e293b", color="#475569"),
        yaxis=dict(title="Score", range=[0, 1], gridcolor="#1e293b", color="#475569"),
    )
    return fig


# ─── Carregamento e Treinamento ───────────────────────────────────────────────

df_hd_raw, df_resp_raw = carregar_dados()
pipeline_hd, pipeline_resp = treinar_modelos(df_hd_raw, df_resp_raw)

# Inferência nos dados completos
df_hd_inf   = pipeline_hd.inferir(df_hd_raw)
df_resp_inf  = pipeline_resp.inferir(df_resp_raw)

# Snapshots (estado atual de cada máquina)
snap_hd   = snapshot_atual(df_hd_inf)
snap_resp = snapshot_atual(df_resp_inf)
snap_all  = pd.concat([snap_hd, snap_resp], ignore_index=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 ManuIA")
    st.markdown("**Sistema de Manutenção Preditiva**")
    st.markdown("---")

    pagina = st.radio(
        "Navegação",
        ["📊 Visão Geral", "🩸 Hemodiálise", "💨 Respiradores", "🤖 Modelos de IA"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Atualizar simulação**")
    if st.button("🔄 Regenerar Dados", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("FAESA – Eng. Computação III – 2026")
    st.caption("Dados: NASA C-MAPSS (simulado)")


# ─── Cabeçalho ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>Sistema de Manutenção Preditiva com Inteligência Artificial</h1>
  <p>Hemodiálise & Respiradores Pulmonares · Isolation Forest + Random Forest · NASA C-MAPSS Format</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ═══════════════════════════════════════════════════════════════════════════════

if pagina == "📊 Visão Geral":

    # KPIs
    total_equip = len(snap_all)
    criticos    = (snap_all["urgencia"] == "Crítico").sum()
    alertas     = (snap_all["urgencia"] == "Alerta").sum()
    anomalias   = snap_all["anomalia"].sum() if "anomalia" in snap_all else 0
    rul_medio   = int(snap_all["rul_predito"].mean()) if "rul_predito" in snap_all else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de Equipamentos", total_equip)
    c2.metric("🔴 Críticos",  criticos,  delta=f"{criticos} urgentes", delta_color="inverse")
    c3.metric("🟠 Em Alerta", alertas,   delta=f"{alertas} pendentes", delta_color="inverse")
    c4.metric("⚠️ Anomalias",  int(anomalias))
    c5.metric("RUL Médio",    f"{rul_medio} ciclos")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            grafico_distribuicao_status(snap_hd, "Status — Hemodiálise"),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            grafico_distribuicao_status(snap_resp, "Status — Respiradores"),
            use_container_width=True,
        )

    # Tabela geral
    st.markdown('<p class="section-title">Estado Atual de Todos os Equipamentos</p>',
                unsafe_allow_html=True)

    exibir = snap_all[["maquina_id", "tipo", "ciclo", "rul_predito",
                        "score_anomalia", "urgencia"]].copy()
    exibir["score_anomalia"] = exibir["score_anomalia"].round(3)
    exibir["rul_predito"]    = exibir["rul_predito"].astype(int)

    exibir_sorted = exibir.sort_values(
        ["urgencia"],
        key=lambda x: x.map({"Crítico": 0, "Alerta": 1, "Atenção": 2, "Normal": 3}),
    )

    rows_html = ""
    for _, row in exibir_sorted.iterrows():
        rows_html += f"""
        <tr>
          <td><b>{row['maquina_id']}</b></td>
          <td>{row['tipo']}</td>
          <td>{row['ciclo']}</td>
          <td style="font-family:monospace">{row['rul_predito']}</td>
          <td style="font-family:monospace">{row['score_anomalia']:.3f}</td>
          <td>{badge(row['urgencia'])}</td>
        </tr>"""

    st.markdown(f"""
    <table class="equip-table">
      <thead><tr>
        <th>ID</th><th>Tipo</th><th>Ciclo</th>
        <th>RUL Predito</th><th>Score Anomalia</th><th>Status</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — HEMODIÁLISE
# ═══════════════════════════════════════════════════════════════════════════════

elif pagina == "🩸 Hemodiálise":

    maquinas_hd = sorted(snap_hd["maquina_id"].unique())
    sel = st.selectbox("Selecione a Máquina", maquinas_hd)

    row = snap_hd[snap_hd["maquina_id"] == sel].iloc[0]
    hist = historico_maquina(df_hd_inf, sel)

    # KPIs da máquina
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclo Atual",      int(row["ciclo"]))
    c2.metric("RUL Real",         int(row["rul"]))
    c3.metric("RUL Predito",      int(row["rul_predito"]))
    c4.metric("Score Anomalia",   f"{row['score_anomalia']:.3f}")

    # Gauge
    col_g, col_s = st.columns([1, 2])
    with col_g:
        st.plotly_chart(
            gauge_rul(row["rul_predito"], max_rul=350, titulo=f"RUL — {sel}"),
            use_container_width=True,
        )
    with col_s:
        fig_sc = grafico_anomalia_score(hist, f"Score de Anomalia — {sel}")
        if fig_sc:
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Histórico dos Sensores</p>',
                unsafe_allow_html=True)

    sensores_hd_plot = [
        ("temperatura_sangue",    "Temperatura do Sangue",    "°C"),
        ("pressao_arterial",      "Pressão Arterial",         "mmHg"),
        ("pressao_venosa",        "Pressão Venosa",           "mmHg"),
        ("fluxo_sanguineo",       "Fluxo Sanguíneo",          "mL/min"),
        ("vibracao",              "Vibração da Bomba",         "mm/s"),
        ("temperatura_motor",     "Temperatura do Motor",      "°C"),
    ]

    for i in range(0, len(sensores_hd_plot), 2):
        ca, cb = st.columns(2)
        with ca:
            col, titulo, unid = sensores_hd_plot[i]
            st.plotly_chart(grafico_sensor_historico(hist, col, titulo, unid),
                            use_container_width=True)
        if i + 1 < len(sensores_hd_plot):
            with cb:
                col, titulo, unid = sensores_hd_plot[i + 1]
                st.plotly_chart(grafico_sensor_historico(hist, col, titulo, unid),
                                use_container_width=True)

    # Tabela de últimas leituras
    st.markdown('<p class="section-title">Últimas 10 Leituras</p>',
                unsafe_allow_html=True)
    ultimas = hist.tail(10)[["ciclo", "timestamp", "temperatura_sangue",
                              "pressao_arterial", "pressao_venosa",
                              "fluxo_sanguineo", "vibracao", "temperatura_motor",
                              "anomalia", "score_anomalia"]].copy()
    ultimas["timestamp"] = ultimas["timestamp"].dt.strftime("%d/%m %H:%M")
    ultimas["score_anomalia"] = ultimas["score_anomalia"].round(3)
    st.dataframe(ultimas, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — RESPIRADORES
# ═══════════════════════════════════════════════════════════════════════════════

elif pagina == "💨 Respiradores":

    maquinas_resp = sorted(snap_resp["maquina_id"].unique())
    sel = st.selectbox("Selecione o Respirador", maquinas_resp)

    row = snap_resp[snap_resp["maquina_id"] == sel].iloc[0]
    hist = historico_maquina(df_resp_inf, sel)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclo Atual",    int(row["ciclo"]))
    c2.metric("RUL Real",       int(row["rul"]))
    c3.metric("RUL Predito",    int(row["rul_predito"]))
    c4.metric("Score Anomalia", f"{row['score_anomalia']:.3f}")

    col_g, col_s = st.columns([1, 2])
    with col_g:
        st.plotly_chart(
            gauge_rul(row["rul_predito"], max_rul=300, titulo=f"RUL — {sel}"),
            use_container_width=True,
        )
    with col_s:
        fig_sc = grafico_anomalia_score(hist, f"Score de Anomalia — {sel}")
        if fig_sc:
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Histórico dos Sensores</p>',
                unsafe_allow_html=True)

    sensores_resp_plot = [
        ("pressao_pico",         "Pressão de Pico",         "cmH2O"),
        ("volume_corrente",      "Volume Corrente",          "mL"),
        ("frequencia_resp",      "Frequência Respiratória",  "rpm"),
        ("fio2",                 "FiO2",                     "%"),
        ("vibracao",             "Vibração do Compressor",   "mm/s"),
        ("temperatura_compressor","Temp. Compressor",        "°C"),
    ]

    for i in range(0, len(sensores_resp_plot), 2):
        ca, cb = st.columns(2)
        with ca:
            col, titulo, unid = sensores_resp_plot[i]
            st.plotly_chart(grafico_sensor_historico(hist, col, titulo, unid),
                            use_container_width=True)
        if i + 1 < len(sensores_resp_plot):
            with cb:
                col, titulo, unid = sensores_resp_plot[i + 1]
                st.plotly_chart(grafico_sensor_historico(hist, col, titulo, unid),
                                use_container_width=True)

    st.markdown('<p class="section-title">Últimas 10 Leituras</p>',
                unsafe_allow_html=True)
    ultimas = hist.tail(10)[["ciclo", "timestamp", "pressao_pico",
                              "volume_corrente", "frequencia_resp", "fio2",
                              "vibracao", "temperatura_compressor",
                              "anomalia", "score_anomalia"]].copy()
    ultimas["timestamp"] = ultimas["timestamp"].dt.strftime("%d/%m %H:%M")
    ultimas["score_anomalia"] = ultimas["score_anomalia"].round(3)
    st.dataframe(ultimas, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MODELOS DE IA
# ═══════════════════════════════════════════════════════════════════════════════

elif pagina == "🤖 Modelos de IA":

    st.markdown("### Desempenho dos Modelos de Machine Learning")
    st.markdown("""
    O sistema utiliza dois algoritmos treinados sobre dados no formato **NASA C-MAPSS**:
    - **Isolation Forest** — detecção de anomalias (não supervisionado)  
    - **Random Forest Regressor** — predição de RUL (supervisionado)
    """)

    tab1, tab2 = st.tabs(["Hemodiálise", "Respiradores"])

    for tab, pipeline, df_inf, label in [
        (tab1, pipeline_hd, df_hd_inf, "Hemodiálise"),
        (tab2, pipeline_resp, df_resp_inf, "Respiradores"),
    ]:
        with tab:
            metricas = pipeline.metricas_rul
            c1, c2 = st.columns(2)
            c1.metric("MAE (RUL)", f"{metricas.get('MAE', '—')} ciclos",
                      help="Mean Absolute Error — erro médio absoluto na predição de RUL")
            c2.metric("R² (ajuste)", metricas.get('R²', '—'),
                      help="Coeficiente de determinação — 1.0 = predição perfeita")

            col_a, col_b = st.columns(2)
            with col_a:
                imp = pipeline.preditor.importancia_features()
                st.plotly_chart(grafico_importancia(imp), use_container_width=True)
            with col_b:
                snap = snap_hd if label == "Hemodiálise" else snap_resp
                ultimos_ciclos = df_inf.sort_values("ciclo").groupby("maquina_id").tail(1)
                st.plotly_chart(grafico_rul_scatter(ultimos_ciclos), use_container_width=True)

            st.markdown('<p class="section-title">Distribuição de Anomalias por Máquina</p>',
                        unsafe_allow_html=True)
            anom_por_maq = df_inf.groupby("maquina_id")["anomalia"].mean().reset_index()
            anom_por_maq.columns = ["Máquina", "Taxa de Anomalia"]
            anom_por_maq["Taxa de Anomalia (%)"] = (anom_por_maq["Taxa de Anomalia"] * 100).round(2)
            anom_por_maq = anom_por_maq.sort_values("Taxa de Anomalia (%)", ascending=False)

            fig_anom = go.Figure(go.Bar(
                x=anom_por_maq["Máquina"],
                y=anom_por_maq["Taxa de Anomalia (%)"],
                marker_color=[
                    "#f87171" if v > 10 else "#fb923c" if v > 5 else "#4ade80"
                    for v in anom_por_maq["Taxa de Anomalia (%)"]
                ],
                text=[f"{v:.1f}%" for v in anom_por_maq["Taxa de Anomalia (%)"]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11),
            ))
            fig_anom.update_layout(
                height=280,
                margin=dict(t=20, b=40, l=40, r=20),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(gridcolor="#1e293b", color="#475569"),
                yaxis=dict(title="Taxa (%)", gridcolor="#1e293b", color="#475569"),
                showlegend=False,
            )
            st.plotly_chart(fig_anom, use_container_width=True)

            st.markdown('<p class="section-title">Código do Modelo (Isolation Forest)</p>',
                        unsafe_allow_html=True)
            st.code(f"""
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. Carregamento dos dados (formato NASA C-MAPSS) ---
# Fonte: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
df = pd.read_csv('dados_{label.lower().replace("á","a").replace("é","e")}.csv')

sensores = {pipeline.sensores}

# --- 2. Isolation Forest — Detecção de Anomalias ---
scaler = StandardScaler()
X = scaler.fit_transform(df[sensores])

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # 5% de anomalias esperadas
    random_state=42,
)
iso_forest.fit(X)
df['anomalia'] = iso_forest.predict(X)          # +1 normal, -1 anomalia
df['score']    = iso_forest.score_samples(X)    # score de confiança

# --- 3. Random Forest — Predição de RUL ---
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
rf.fit(scaler.transform(df[sensores]), df['rul'])
df['rul_predito'] = rf.predict(scaler.transform(df[sensores]))

print(df[['maquina_id', 'ciclo', 'rul', 'rul_predito', 'anomalia']].tail())
""", language="python")
