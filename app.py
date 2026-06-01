"""
Dashboard — Pronóstico Precio de Bolsa Colombia
Ejecutar: streamlit run app.py
Datos reales: XM (precio, aportes, embalses, demanda) + NOAA (ONI)
"""

import io
import os
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from datetime import date

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Precio de Bolsa · Colombia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .kpi-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center; }
    .kpi-label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #7d8590; margin-bottom: 4px; }
    .kpi-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700; color: #e6edf3; }
    .kpi-delta-up   { color: #f85149; font-size: 13px; }
    .kpi-delta-down { color: #3fb950; font-size: 13px; }
    .kpi-delta-flat { color: #7d8590; font-size: 13px; }
    .alerta-alto   { background:#2d1b1b; border-left:3px solid #f85149; border-radius:6px; padding:10px 14px; color:#f85149; font-size:14px; }
    .alerta-bajo   { background:#1b2d1b; border-left:3px solid #3fb950; border-radius:6px; padding:10px 14px; color:#3fb950; font-size:14px; }
    .alerta-normal { background:#1b1f2d; border-left:3px solid #58a6ff; border-radius:6px; padding:10px 14px; color:#58a6ff; font-size:14px; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] .stMarkdown p { color: #7d8590; font-size: 13px; }
    div[data-testid="metric-container"] { display:none; }
    .stPlotlyChart { border-radius: 12px; overflow: hidden; }
    .fuente-badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px; font-weight:500; margin-left:8px; }
    .fuente-real { background:#1b2d1b; color:#3fb950; border:1px solid #3fb950; }
    .fuente-sim  { background:#1b1f2d; color:#7d8590; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "bg": "#0d1117", "surface": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "muted": "#7d8590", "blue": "#58a6ff",
    "orange": "#f0883e", "green": "#3fb950", "red": "#f85149",
}

# ─────────────────────────────────────────────
# FUNCIONES DE CARGA DE DATOS
# ─────────────────────────────────────────────

def aplanar_xm(df_xm, nombre):
    """
    pydataxm devuelve columnas: Date, Values_Hour01 ... Values_Hour24
    Promediamos las 24 horas para tener un valor diario.
    """
    cols_hora = [c for c in df_xm.columns if c.startswith("Values_")]
    df_xm = df_xm.copy()
    df_xm[nombre] = df_xm[cols_hora].mean(axis=1)
    return df_xm[["Date", nombre]].rename(columns={"Date": "fecha"})


@st.cache_data(ttl=3600)
def cargar_precio_excel():
    """
    Lee el precio de bolsa desde el Excel descargado de XM.
    Estructura: fila 1 vacía, fila 2 título, fila 3 headers (Fecha, 0..23, Version)
    """
    ruta = os.path.join(os.path.dirname(__file__), "PrecioBolsa2026.xlsx")
    df_raw = pd.read_excel(ruta, sheet_name="PrecioBolsa", header=2, parse_dates=["Fecha"])
    df_raw = df_raw.rename(columns={"Fecha": "fecha"})
    cols_horas = [str(h) for h in range(24)]
    cols_horas = [c for c in cols_horas if c in df_raw.columns]
    df_raw["precio"] = df_raw[cols_horas].mean(axis=1)
    df = df_raw[["fecha", "precio"]].dropna()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


@st.cache_data(ttl=3600)
def cargar_xm_variables(fecha_inicio, fecha_fin):
    """
    Descarga aportes, embalses y demanda desde la API pública de XM.
    No requiere credenciales.
    Variables:
      - AporEner    : Aportes energéticos hídricos (GWh/día)
      - NivEmbBalse : Nivel de embalses (%)
      - DemaSistNaci: Demanda nacional (GWh/día)
    """
    try:
        from pydataxm import ReadDB
    except ImportError:
        try:
            from pydataxm.pydataxm import ReadDB
        except ImportError:
            return {}, ["aportes", "embalses", "demanda"]

    try:
        obj = ReadDB()
    except Exception:
        return {}, ["aportes", "embalses", "demanda"]

    resultados = {}
    variables = {
        "aportes":  ("AporEner",     "Sistema"),
        "embalses": ("NivEmbBalse",  "Sistema"),
        "demanda":  ("DemaSistNaci", "Sistema"),
    }
    errores = []
    for nombre, (variable, entidad) in variables.items():
        try:
            df_raw = obj.request_data(variable, entidad, fecha_inicio, fecha_fin)
            if df_raw is not None and len(df_raw) > 0:
                resultados[nombre] = aplanar_xm(df_raw, nombre)
            else:
                errores.append(nombre)
        except Exception:
            errores.append(nombre)
    return resultados, errores


@st.cache_data(ttl=86400)  # cache 24h — el ONI se actualiza mensualmente
def cargar_oni():
    """
    Descarga el índice ONI de la NOAA y lo convierte a serie diaria.
    URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
    Formato: columnas SEAS | YR | ONI
    Cada fila es un trimestre móvil (DJF, JFM, FMA, ..., NDJ)
    """
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df_oni = pd.read_csv(
            io.StringIO(resp.text),
            sep=r"\s+",
            skiprows=1,
            names=["trimestre", "anio", "oni"],
        )
        # Mes central de cada trimestre móvil
        mes_central = {
            "DJF": 1,  "JFM": 2,  "FMA": 3,  "MAM": 4,
            "AMJ": 5,  "MJJ": 6,  "JJA": 7,  "JAS": 8,
            "ASO": 9,  "SON": 10, "OND": 11, "NDJ": 12,
        }
        df_oni["mes"] = df_oni["trimestre"].map(mes_central)
        df_oni = df_oni.dropna(subset=["mes", "oni"])
        df_oni["fecha"] = pd.to_datetime(
            df_oni["anio"].astype(int).astype(str) + "-" +
            df_oni["mes"].astype(int).astype(str).str.zfill(2) + "-01"
        )
        df_oni = df_oni[["fecha", "oni"]].sort_values("fecha")

        # Expandir a serie diaria por forward-fill mensual
        fechas = pd.date_range(df_oni["fecha"].min(),
                               date.today(), freq="D")
        df_diario = pd.DataFrame({"fecha": fechas})
        df_diario = df_diario.merge(df_oni, on="fecha", how="left")
        df_diario["oni"] = df_diario["oni"].ffill()
        return df_diario.set_index("fecha")["oni"], True
    except Exception:
        return None, False


@st.cache_data(ttl=3600)
def cargar_datos():
    """
    Ensambla el dataset completo:
    - Precio: Excel XM (real)
    - Aportes, embalses, demanda: API pydataxm (real cuando disponible)
    - ONI: NOAA API (real cuando disponible)
    Fallback a simulación si alguna fuente falla.
    """
    df = cargar_precio_excel()
    fecha_inicio = df["fecha"].min().date()
    fecha_fin    = df["fecha"].max().date()
    n = len(df)
    t = np.arange(n)
    est = 30 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    np.random.seed(42)

    fuentes = {"precio": "real"}

    # ── Variables XM ──
    xm_vars, errores = cargar_xm_variables(fecha_inicio, fecha_fin)

    for nombre in ["aportes", "embalses", "demanda"]:
        if nombre in xm_vars:
            df = df.merge(xm_vars[nombre], on="fecha", how="left")
            fuentes[nombre] = "real"
        else:
            fuentes[nombre] = "simulado"

    # Fallback simulado para variables no disponibles
    if "aportes" not in df.columns or df["aportes"].isna().all():
        df["aportes"] = np.round(np.clip(3500 - 20*est + np.random.normal(0, 200, n), 500, 7000), 1)
        fuentes["aportes"] = "simulado"
    if "embalses" not in df.columns or df["embalses"].isna().all():
        df["embalses"] = np.round(np.clip(65 - 0.3*est + np.random.normal(0, 5, n), 10, 100), 1)
        fuentes["embalses"] = "simulado"
    if "demanda" not in df.columns or df["demanda"].isna().all():
        df["demanda"] = np.round(165 + 0.005*t + 5*np.sin(2*np.pi*t/365) + np.random.normal(0, 3, n), 1)
        fuentes["demanda"] = "simulado"

    # Rellenar NaN residuales con interpolación
    for col in ["aportes", "embalses", "demanda"]:
        df[col] = df[col].interpolate().ffill().bfill()

    # ── ONI ──
    oni_serie, oni_ok = cargar_oni()
    if oni_ok and oni_serie is not None:
        df["oni"] = df["fecha"].map(oni_serie).ffill().bfill().fillna(0.0)
        fuentes["oni"] = "real"
    else:
        df["oni"] = 0.0
        fuentes["oni"] = "simulado"

    df = df.sort_values("fecha").reset_index(drop=True)
    return df, fuentes


@st.cache_data(ttl=3600)
def entrenar_y_pronosticar(horizonte_dias: int):
    df, fuentes = cargar_datos()

    dp = df.rename(columns={"fecha": "ds", "precio": "y"}).copy()

    # Normalizar regresores — manejar std=0 (columna constante)
    stats = {}
    for col in ["aportes", "embalses", "demanda", "oni"]:
        mu  = dp[col].mean()
        std = dp[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        dp[f"{col}_norm"] = (dp[col] - mu) / std
        stats[col] = (mu, std)

    # Quitar regresores con NaN después de normalizar
    regresores = []
    for col in ["aportes", "embalses", "demanda", "oni"]:
        col_norm = f"{col}_norm"
        if dp[col_norm].isna().sum() == 0:
            regresores.append(col_norm)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    prior_scales = {
        "aportes_norm": 0.5,
        "embalses_norm": 0.4,
        "demanda_norm": 0.3,
        "oni_norm": 0.6,
    }
    for reg in regresores:
        m.add_regressor(reg, prior_scale=prior_scales.get(reg, 0.5))

    m.fit(dp)

    futuro = m.make_future_dataframe(periods=horizonte_dias, freq="D")
    ultima = dp["ds"].max()

    for col_norm in regresores:
        hist_rec = dp[dp["ds"] >= ultima - pd.Timedelta(days=90)][col_norm]
        mu_r  = hist_rec.mean()
        std_r = max(hist_rec.std() * 0.3, 0.01)
        proy  = np.random.normal(mu_r, std_r, horizonte_dias)
        futuro[col_norm] = np.concatenate([dp[col_norm].values, proy])

    forecast = m.predict(futuro)
    return df, dp, forecast, m, fuentes, regresores


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Configuración")
    st.markdown("---")
    horizonte        = st.slider("Horizonte de pronóstico (días)", 7, 90, 30, step=7)
    umbral_alto_pct  = st.slider("Percentil alerta alta", 70, 95, 85, step=5)
    umbral_bajo_pct  = st.slider("Percentil alerta baja", 5, 30, 15, step=5)
    mostrar_comp     = st.checkbox("Mostrar componentes del modelo", value=False)
    mostrar_tabla    = st.checkbox("Mostrar tabla de pronóstico", value=True)
    st.markdown("---")
    st.markdown("*Dashboard · Prophet + Streamlit*")
    st.markdown("*Datos: XM + NOAA*")


# ─────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:700;color:#e6edf3;margin-bottom:0;'>
    ⚡ Precio de Bolsa · Colombia
</h1>
<p style='color:#7d8590;font-size:14px;margin-top:4px;'>
    Prophet · XM (precio, aportes, embalses, demanda) · NOAA (ONI/ENSO)
</p>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# CARGA Y ENTRENAMIENTO
# ─────────────────────────────────────────────

with st.spinner("Cargando datos y entrenando modelo..."):
    df, dp, forecast, modelo, fuentes, regresores = entrenar_y_pronosticar(horizonte)

corte        = dp["ds"].max()
futuro_fc    = forecast[forecast["ds"] > corte].copy()
hist_fc      = forecast[forecast["ds"] <= corte].copy()
umbral_alto  = np.percentile(dp["y"], umbral_alto_pct)
umbral_bajo  = np.percentile(dp["y"], umbral_bajo_pct)
precio_hoy   = dp["y"].iloc[-1]
precio_fc_d1 = futuro_fc["yhat"].iloc[0]  if len(futuro_fc) > 0 else precio_hoy
precio_fc_fn = futuro_fc["yhat"].iloc[-1] if len(futuro_fc) > 0 else precio_hoy
delta_pct    = (precio_fc_d1 - precio_hoy) / precio_hoy * 100
alerta_nivel = "ALTO" if precio_fc_d1 > umbral_alto else ("BAJO" if precio_fc_d1 < umbral_bajo else "NORMAL")

# Badges de fuente de datos
def badge(nombre):
    es_real = fuentes.get(nombre) == "real"
    cls  = "fuente-real" if es_real else "fuente-sim"
    txt  = "✓ real" if es_real else "≈ simulado"
    return f'<span class="fuente-badge {cls}">{txt}</span>'

st.markdown(
    f"**Estado de fuentes:** "
    f"Precio {badge('precio')} &nbsp; "
    f"Aportes {badge('aportes')} &nbsp; "
    f"Embalses {badge('embalses')} &nbsp; "
    f"Demanda {badge('demanda')} &nbsp; "
    f"ONI {badge('oni')}",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

def kpi(label, value, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls  = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.1f}% {delta_label}</div>'
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{delta_html}</div>'

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(kpi("Precio actual",           f"${precio_hoy:.0f}"),                         unsafe_allow_html=True)
with c2: st.markdown(kpi("Pronóstico mañana",       f"${precio_fc_d1:.0f}", delta=delta_pct, delta_label="vs hoy"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"Pronóstico día {horizonte}", f"${precio_fc_fn:.0f}"),                   unsafe_allow_html=True)
with c4: st.markdown(kpi("Nivel embalses",          f"{df['embalses'].iloc[-1]:.0f}%"),             unsafe_allow_html=True)
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
# TABS PRINCIPALES
# ─────────────────────────────────────────────

tab1, tab2 = st.tabs(["📈 Pronóstico", "🔍 Variables externas"])

with tab1:
    dias_hist = st.select_slider(
        "Histórico a mostrar",
        options=[90, 180, 365, 730, 1825],
        value=90 if len(dp) < 180 else (180 if len(dp) < 365 else 365),
        format_func=lambda x: f"{x//365}a" if x >= 365 else f"{x}d",
    )
    corte_hist = corte - pd.Timedelta(days=int(dias_hist))
    dh = dp[dp["ds"] >= corte_hist]
    fh = hist_fc[hist_fc["ds"] >= corte_hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([fh["ds"], fh["ds"][::-1]]),
        y=pd.concat([fh["yhat_upper"], fh["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([futuro_fc["ds"], futuro_fc["ds"][::-1]]),
        y=pd.concat([futuro_fc["yhat_upper"], futuro_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(240,136,62,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Intervalo 80%",
    ))
    fig.add_trace(go.Scatter(x=dh["ds"], y=dh["y"],
        mode="lines", line=dict(color=COLORS["muted"], width=1), name="Real"))
    fig.add_trace(go.Scatter(x=fh["ds"], y=fh["yhat"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5, dash="dot"),
        name="Ajuste modelo", opacity=0.7))
    fig.add_trace(go.Scatter(x=futuro_fc["ds"], y=futuro_fc["yhat"],
        mode="lines", line=dict(color=COLORS["orange"], width=2.5),
        name=f"Pronóstico {horizonte}d"))
    fig.add_hline(y=umbral_alto, line=dict(color=COLORS["red"],   width=1, dash="dot"),
                  annotation_text=f"Alerta alta · ${umbral_alto:.0f}", annotation_font_color=COLORS["red"])
    fig.add_hline(y=umbral_bajo, line=dict(color=COLORS["green"], width=1, dash="dot"),
                  annotation_text=f"Alerta baja · ${umbral_bajo:.0f}", annotation_font_color=COLORS["green"])
    fig.add_trace(go.Scatter(
        x=[corte, corte], y=[dp["y"].min()*0.9, dp["y"].max()*1.1],
        mode="lines+text", line=dict(color="#ffffff", width=1, dash="dash"),
        text=["", "Hoy"], textposition="top center",
        textfont=dict(color="#ffffff", size=11), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
        font=dict(family="DM Sans", color=COLORS["text"]),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="$/kWh", gridcolor=COLORS["border"], zeroline=False),
        xaxis=dict(gridcolor=COLORS["border"]),
        height=420, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if mostrar_comp:
        st.markdown("**Componentes del modelo**")
        st.pyplot(modelo.plot_components(forecast))


with tab2:
    col_a, col_b = st.columns(2)

    def mini_line(df_plot, x_col, y_col, color, title, ylabel, fuente_nombre):
        es_real = fuentes.get(fuente_nombre) == "real"
        titulo_completo = title + (" ✓" if es_real else " ≈")
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=df_plot[x_col], y=df_plot[y_col],
            mode="lines", line=dict(color=color, width=1.5), fill="tozeroy",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
        ))
        f.update_layout(
            title=dict(text=titulo_completo, font=dict(size=13, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(family="DM Sans", color=COLORS["text"]),
            yaxis=dict(title=ylabel, gridcolor=COLORS["border"], zeroline=False),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=260, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        )
        return f

    corte_730 = corte - pd.Timedelta(days=730)
    df_full = df[df["fecha"] >= corte_730].copy()
    df_full_ds = df_full.rename(columns={"fecha": "ds"})

    with col_a:
        st.plotly_chart(mini_line(df_full_ds, "ds", "aportes", "#58a6ff",
                                  "Aportes hídricos", "GWh/día", "aportes"),
                        use_container_width=True)
        st.plotly_chart(mini_line(df_full_ds, "ds", "demanda", "#bc8cff",
                                  "Demanda nacional", "GWh/día", "demanda"),
                        use_container_width=True)

    with col_b:
        st.plotly_chart(mini_line(df_full_ds, "ds", "embalses", "#3fb950",
                                  "Nivel embalses", "%", "embalses"),
                        use_container_width=True)

        # ONI — barras con color según signo
        es_oni_real = fuentes.get("oni") == "real"
        titulo_oni  = "Índice ONI/ENSO ✓" if es_oni_real else "Índice ONI/ENSO ≈"
        fig_oni = go.Figure()
        fig_oni.add_trace(go.Bar(
            x=df_full_ds["ds"],
            y=df_full_ds["oni"],
            marker_color=["#f85149" if v > 0 else "#58a6ff"
                          for v in df_full_ds["oni"]],
            name="ONI",
        ))
        fig_oni.update_layout(
            title=dict(text=titulo_oni, font=dict(size=13, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(family="DM Sans", color=COLORS["text"]),
            yaxis=dict(title="°C anomalía", gridcolor=COLORS["border"],
                       zeroline=True, zerolinecolor=COLORS["border"]),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=260, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        )
        st.plotly_chart(fig_oni, use_container_width=True)

    # Info de regresores activos
    st.caption(f"Regresores activos en el modelo: {', '.join(regresores)}")


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
    st.download_button("⬇ Descargar CSV", data=csv,
                       file_name=f"pronostico_energia_{date.today()}.csv",
                       mime="text/csv")


# ─────────────────────────────────────────────
# MÉTRICAS DEL MODELO
# ─────────────────────────────────────────────

with st.expander("📊 Métricas del modelo"):
    residuales = dp["y"].values - hist_fc["yhat"].values[:len(dp)]
    mape = np.mean(np.abs(residuales / dp["y"].values)) * 100
    rmse = np.sqrt(np.mean(residuales ** 2))
    mae  = np.mean(np.abs(residuales))
    m1, m2, m3 = st.columns(3)
    m1.metric("MAPE (in-sample)", f"{mape:.1f}%")
    m2.metric("RMSE", f"{rmse:.1f} $/kWh")
    m3.metric("MAE",  f"{mae:.1f} $/kWh")
    st.caption(
        "Métricas in-sample (el modelo vio estos datos). "
        "Para evaluación real usar cross_validation() del archivo modelo_prophet_energia.py."
    )
