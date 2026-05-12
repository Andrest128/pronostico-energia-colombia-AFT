"""
Dashboard — Pronóstico Precio de Bolsa Colombia
Ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Pronóstico Precio de Bolsa · Colombia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Tarjetas KPI */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7d8590;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
    }
    .kpi-delta-up   { color: #f85149; font-size: 13px; }
    .kpi-delta-down { color: #3fb950; font-size: 13px; }
    .kpi-delta-flat { color: #7d8590; font-size: 13px; }

    /* Alertas */
    .alerta-alto   { background:#2d1b1b; border-left:3px solid #f85149; border-radius:6px; padding:10px 14px; color:#f85149; font-size:14px; }
    .alerta-bajo   { background:#1b2d1b; border-left:3px solid #3fb950; border-radius:6px; padding:10px 14px; color:#3fb950; font-size:14px; }
    .alerta-normal { background:#1b1f2d; border-left:3px solid #58a6ff; border-radius:6px; padding:10px 14px; color:#58a6ff; font-size:14px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] .stMarkdown p { color: #7d8590; font-size: 13px; }

    div[data-testid="metric-container"] { display:none; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cargar_datos():
    """
    Genera datos simulados. Reemplazar con:
        from pydataxm import ReadDB
        obj = ReadDB()
        precio = obj.request_data("PrecBolsNaci", "Sistema", date(2015,1,1), date(2025,3,31))
    """
    fechas = pd.date_range("2015-01-01", "2025-03-31", freq="D")
    n = len(fechas)
    np.random.seed(42)
    t = np.arange(n)

    est = 30 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    nino = np.zeros(n)
    for i, f in enumerate(fechas):
        if pd.Timestamp("2015-06-01") <= f <= pd.Timestamp("2016-05-31"):
            nino[i] = 80
        elif pd.Timestamp("2023-07-01") <= f <= pd.Timestamp("2024-03-31"):
            nino[i] = 50

    precio   = np.clip(180 + 0.01*t + est + 0.8*nino + np.random.normal(0, 15, n), 50, 600)
    aportes  = np.clip(3500 - 20*est - 0.4*nino + np.random.normal(0, 200, n), 500, 7000)
    embalses = np.clip(65 - 0.3*est - 0.2*nino + np.random.normal(0, 5, n), 10, 100)
    demanda  = 165 + 0.005*t + 5*np.sin(2*np.pi*t/365) + np.random.normal(0, 3, n)
    oni = np.zeros(n)
    for i, f in enumerate(fechas):
        if pd.Timestamp("2015-06-01") <= f <= pd.Timestamp("2016-05-31"):  oni[i] =  2.3
        elif pd.Timestamp("2020-08-01") <= f <= pd.Timestamp("2021-04-30"): oni[i] = -1.2
        elif pd.Timestamp("2023-07-01") <= f <= pd.Timestamp("2024-03-31"): oni[i] =  1.8

    return pd.DataFrame({
        "fecha": fechas,
        "precio": np.round(precio, 2),
        "aportes": np.round(aportes, 1),
        "embalses": np.round(embalses, 1),
        "demanda": np.round(demanda, 1),
        "oni": np.round(oni, 2),
    })


@st.cache_data(ttl=3600)
def entrenar_y_pronosticar(horizonte_dias: int):
    df = cargar_datos()

    dp = df.rename(columns={"fecha": "ds", "precio": "y"}).copy()
    for col in ["aportes", "embalses", "demanda", "oni"]:
        mu, std = dp[col].mean(), dp[col].std()
        dp[f"{col}_norm"] = (dp[col] - mu) / std

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    for reg in ["aportes_norm", "embalses_norm", "demanda_norm", "oni_norm"]:
        m.add_regressor(reg, prior_scale=0.5)

    m.fit(dp)

    futuro = m.make_future_dataframe(periods=horizonte_dias, freq="D")
    ultima = dp["ds"].max()
    for col in ["aportes_norm", "embalses_norm", "demanda_norm", "oni_norm"]:
        hist_rec = dp[dp["ds"] >= ultima - pd.Timedelta(days=90)][col]
        mu_r, std_r = hist_rec.mean(), hist_rec.std() * 0.3
        proy = np.random.normal(mu_r, std_r, horizonte_dias)
        futuro[col] = np.concatenate([dp[col].values, proy])

    forecast = m.predict(futuro)
    return df, dp, forecast, m


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Configuración")
    st.markdown("---")

    horizonte = st.slider("Horizonte de pronóstico (días)", 7, 90, 30, step=7)
    umbral_alto_pct = st.slider("Percentil alerta alta", 70, 95, 85, step=5)
    umbral_bajo_pct = st.slider("Percentil alerta baja", 5, 30, 15, step=5)
    mostrar_componentes = st.checkbox("Mostrar componentes del modelo", value=False)
    mostrar_tabla = st.checkbox("Mostrar tabla de pronóstico", value=True)

    st.markdown("---")
    st.markdown("**Fuente de datos**")
    fuente = st.radio("", ["Datos simulados", "API XM (próximamente)"], index=0)

    st.markdown("---")
    st.markdown("*Dashboard desarrollado con Prophet + Streamlit*")


# ─────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2rem; font-weight:700;
           color:#e6edf3; margin-bottom:0;'>
    Pronóstico Precio de Energía en Bolsa &nbsp;·&nbsp; Colombia
</h1>
<p style='color:#7d8590; font-size:14px; margin-top:4px;'>
    Pronóstico basado en Prophet · Variables: hidrología, demanda, ENSO
</p>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────
# CARGA Y CÁLCULO
# ─────────────────────────────────────────────

with st.spinner("Entrenando modelo..."):
    df, dp, forecast, modelo = entrenar_y_pronosticar(horizonte)

corte      = dp["ds"].max()
futuro_fc  = forecast[forecast["ds"] > corte].copy()
hist_fc    = forecast[forecast["ds"] <= corte].copy()

umbral_alto = np.percentile(dp["y"], umbral_alto_pct)
umbral_bajo = np.percentile(dp["y"], umbral_bajo_pct)

precio_hoy    = dp["y"].iloc[-1]
precio_fc_d1  = futuro_fc["yhat"].iloc[0]  if len(futuro_fc) > 0 else precio_hoy
precio_fc_fin = futuro_fc["yhat"].iloc[-1] if len(futuro_fc) > 0 else precio_hoy
delta_pct     = (precio_fc_d1 - precio_hoy) / precio_hoy * 100

alerta_nivel  = "ALTO" if precio_fc_d1 > umbral_alto else ("BAJO" if precio_fc_d1 < umbral_bajo else "NORMAL")


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

def kpi(label, value, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls  = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.1f}% {delta_label}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

with c1:
    st.markdown(kpi("Precio actual", f"${precio_hoy:.0f}", delta_label=""), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Pronóstico mañana", f"${precio_fc_d1:.0f}", delta=delta_pct, delta_label="vs hoy"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi(f"Pronóstico día {horizonte}", f"${precio_fc_fin:.0f}"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi("Nivel embalses", f"{df['embalses'].iloc[-1]:.0f}%"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ALERTA
# ─────────────────────────────────────────────

alertas_map = {
    "ALTO":   ("alerta-alto",   f"⚠ Alerta: precio pronosticado ALTO (>${umbral_alto:.0f}/kWh · p{umbral_alto_pct})"),
    "BAJO":   ("alerta-bajo",   f"✓ Precio pronosticado BAJO (<${umbral_bajo:.0f}/kWh · p{umbral_bajo_pct}) — oportunidad de compra"),
    "NORMAL": ("alerta-normal", f"● Precio pronosticado en rango NORMAL"),
}
css_cls, msg = alertas_map[alerta_nivel]
st.markdown(f'<div class="{css_cls}">{msg}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GRÁFICA PRINCIPAL
# ─────────────────────────────────────────────

COLORS = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#30363d",
    "text":     "#e6edf3",
    "muted":    "#7d8590",
    "blue":     "#58a6ff",
    "orange":   "#f0883e",
    "green":    "#3fb950",
    "red":      "#f85149",
}

tab1, tab2 = st.tabs(["📈 Pronóstico", "🔍 Variables externas"])

with tab1:
    dias_hist = st.select_slider(
        "Histórico a mostrar",
        options=[90, 180, 365, 730, 1825],
        value=365,
        format_func=lambda x: f"{x//365}a" if x >= 365 else f"{x}d",
    )

    corte_hist = corte - pd.Timedelta(days=int(dias_hist))
    mask_hist = dp["ds"] >= corte_hist
    dh = dp[mask_hist]
    fh = hist_fc[hist_fc["ds"] >= corte_hist]

    fig = go.Figure()

    # Intervalo de confianza histórico
    fig.add_trace(go.Scatter(
        x=pd.concat([fh["ds"], fh["ds"][::-1]]),
        y=pd.concat([fh["yhat_upper"], fh["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    # Intervalo de confianza futuro
    fig.add_trace(go.Scatter(
        x=pd.concat([futuro_fc["ds"], futuro_fc["ds"][::-1]]),
        y=pd.concat([futuro_fc["yhat_upper"], futuro_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(240,136,62,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo 80%",
    ))

    # Precio real
    fig.add_trace(go.Scatter(
        x=dh["ds"], y=dh["y"],
        mode="lines", line=dict(color=COLORS["muted"], width=1),
        name="Real",
    ))

    # Pronóstico histórico (fitted)
    fig.add_trace(go.Scatter(
        x=fh["ds"], y=fh["yhat"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5, dash="dot"),
        name="Ajuste modelo", opacity=0.7,
    ))

    # Pronóstico futuro
    fig.add_trace(go.Scatter(
        x=futuro_fc["ds"], y=futuro_fc["yhat"],
        mode="lines", line=dict(color=COLORS["orange"], width=2.5),
        name=f"Pronóstico {horizonte}d",
    ))

    # Umbrales
    fig.add_hline(y=umbral_alto, line=dict(color=COLORS["red"],   width=1, dash="dot"),
                  annotation_text=f"Alerta alta · ${umbral_alto:.0f}", annotation_font_color=COLORS["red"])
    fig.add_hline(y=umbral_bajo, line=dict(color=COLORS["green"], width=1, dash="dot"),
                  annotation_text=f"Alerta baja · ${umbral_bajo:.0f}", annotation_font_color=COLORS["green"])

    # Línea de corte "hoy" (compatible con todas las versiones de Plotly)
    fig.add_trace(go.Scatter(
        x=[corte, corte],
        y=[dp["y"].min() * 0.9, dp["y"].max() * 1.1],
        mode="lines+text",
        line=dict(color="#ffffff", width=1, dash="dash"),
        text=["", "Hoy"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
        font=dict(family="DM Sans", color=COLORS["text"]),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="$/kWh", gridcolor=COLORS["border"], zeroline=False),
        xaxis=dict(gridcolor=COLORS["border"]),
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if mostrar_componentes:
        st.markdown("**Componentes del modelo**")
        comp_fig = modelo.plot_components(forecast)
        st.pyplot(comp_fig)


with tab2:
    col_a, col_b = st.columns(2)

    def mini_chart(df_plot, x_col, y_col, color, title, ylabel):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=df_plot[x_col], y=df_plot[y_col],
            mode="lines", line=dict(color=color, width=1.5), fill="tozeroy",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
        ))
        f.update_layout(
            title=dict(text=title, font=dict(size=13, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(family="DM Sans", color=COLORS["text"]),
            yaxis=dict(title=ylabel, gridcolor=COLORS["border"], zeroline=False),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=260, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        )
        return f

    corte_730 = corte - pd.Timedelta(days=730)
    mask_full = dp["ds"] >= corte_730
    df_full = df[df["fecha"] >= corte_730]

    with col_a:
        st.plotly_chart(
            mini_chart(df_full.rename(columns={"fecha":"ds"}),
                       "ds", "aportes", "#58a6ff", "Aportes hídricos", "GWh"),
            use_container_width=True
        )
        st.plotly_chart(
            mini_chart(df_full.rename(columns={"fecha":"ds"}),
                       "ds", "demanda", "#bc8cff", "Demanda nacional", "GWh/día"),
            use_container_width=True
        )
    with col_b:
        st.plotly_chart(
            mini_chart(df_full.rename(columns={"fecha":"ds"}),
                       "ds", "embalses", "#3fb950", "Nivel embalses", "%"),
            use_container_width=True
        )
        # ONI con colores positivo/negativo
        df_oni = df_full.rename(columns={"fecha":"ds"})
        fig_oni = go.Figure()
        fig_oni.add_trace(go.Bar(
            x=df_oni["ds"], y=df_oni["oni"],
            marker_color=["#f85149" if v > 0 else "#58a6ff" for v in df_oni["oni"]],
            name="ONI",
        ))
        fig_oni.update_layout(
            title=dict(text="Índice ONI (ENSO)", font=dict(size=13, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(family="DM Sans", color=COLORS["text"]),
            yaxis=dict(title="°C anomalía", gridcolor=COLORS["border"], zeroline=True,
                       zerolinecolor=COLORS["border"]),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=260, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        )
        st.plotly_chart(fig_oni, use_container_width=True)


# ─────────────────────────────────────────────
# TABLA + DESCARGA
# ─────────────────────────────────────────────

if mostrar_tabla:
    st.markdown("### Tabla de pronóstico")

    tabla = futuro_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tabla.columns = ["Fecha", "Pronóstico ($/kWh)", "Límite inferior 80%", "Límite superior 80%"]
    tabla["Fecha"] = tabla["Fecha"].dt.strftime("%Y-%m-%d")
    tabla["Alerta"] = tabla["Pronóstico ($/kWh)"].apply(
        lambda p: "🔴 ALTO" if p > umbral_alto else ("🟢 BAJO" if p < umbral_bajo else "🔵 NORMAL")
    )
    for col in ["Pronóstico ($/kWh)", "Límite inferior 80%", "Límite superior 80%"]:
        tabla[col] = tabla[col].round(1)

    st.dataframe(tabla, use_container_width=True, hide_index=True)

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Descargar CSV",
        data=csv,
        file_name=f"pronostico_energia_{date.today()}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# MÉTRICAS DEL MODELO (colapsable)
# ─────────────────────────────────────────────

with st.expander("📊 Métricas del modelo"):
    residuales = dp["y"].values - hist_fc["yhat"].values[:len(dp)]
    mape = np.mean(np.abs(residuales / dp["y"].values)) * 100
    rmse = np.sqrt(np.mean(residuales ** 2))
    mae  = np.mean(np.abs(residuales))

    m1, m2, m3 = st.columns(3)
    m1.metric("MAPE (in-sample)", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
    m2.metric("RMSE", f"{rmse:.1f} $/kWh", help="Root Mean Square Error")
    m3.metric("MAE",  f"{mae:.1f} $/kWh",  help="Mean Absolute Error")

    st.caption("Para métricas out-of-sample (recomendado), ejecutar validación cruzada con `cross_validation()` en `modelo_prophet_energia.py`.")
