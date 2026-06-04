"""
Dashboard — Pronóstico Precio de Bolsa Colombia
Ejecutar: streamlit run app.py

Archivos base (obligatorio + opcionales fijos):
  - PrecioBolsa2026.xlsx  : hoja 'PrecioBolsa'        | Fecha + horas 0..23 ($/kWh)
  - Aportes2026.xlsx      : hoja 'Aportes'             | Fecha + horas 0..23 (Wh)
  - Embalses2026.xlsx     : hoja 'Reservas_Diario_SIN' | Fecha | Volumen Útil Diario %
  - Demanda2026.xlsx      : hoja 'Demanda'             | Fecha + horas 0..23 (Wh)
  - ONI.xlsx              : hoja 'ONI'                 | fecha | oni

Variables extra (cualquier cantidad):
  Sube un Excel con exactamente 2 columnas: fecha | valor
  El nombre del archivo define el nombre de la variable.
  Ejemplos: WTI.xlsx, USDCOP.xlsx, Gas.xlsx, Carbon.xlsx
  El código los detecta automáticamente — no hay que tocar nada más.
"""

import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import date
warnings.filterwarnings("ignore")

DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────
# VARIABLES EXTRA — CONFIGURACIÓN
# ─────────────────────────────────────────────
# Lista de archivos extra que el modelo intentará cargar.
# Formato de cada archivo: 2 columnas → fecha | valor (diario)
# Para agregar una nueva variable: simplemente súbela al repo
# con el nombre correcto y agrégala aquí.
# prior_scale: qué tanto confía el modelo en esa variable (0.1=poco, 1.0=mucho)

VARIABLES_EXTRA = {
    # nombre_col    : (archivo,          prior_scale, unidad_display)
    "wti"          : ("WTI.xlsx",        0.4,         "USD/barril"),
    "usdcop"       : ("USDCOP.xlsx",     0.4,         "COP/USD"),
    "gas"          : ("Gas.xlsx",        0.4,         "USD/MMBTU"),
    "carbon"       : ("Carbon.xlsx",     0.3,         "USD/ton"),
}

# ─────────────────────────────────────────────
# CONFIGURACIÓN UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Precio de Bolsa · Colombia",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .kpi-card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem 1.4rem; text-align:center; }
    .kpi-label { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#7d8590; margin-bottom:4px; }
    .kpi-value { font-family:'Syne',sans-serif; font-size:28px; font-weight:700; color:#e6edf3; }
    .kpi-delta-up   { color:#f85149; font-size:13px; }
    .kpi-delta-down { color:#3fb950; font-size:13px; }
    .kpi-delta-flat { color:#7d8590; font-size:13px; }
    .alerta-alto   { background:#2d1b1b; border-left:3px solid #f85149; border-radius:6px; padding:10px 14px; color:#f85149; font-size:14px; }
    .alerta-bajo   { background:#1b2d1b; border-left:3px solid #3fb950; border-radius:6px; padding:10px 14px; color:#3fb950; font-size:14px; }
    .alerta-normal { background:#1b1f2d; border-left:3px solid #58a6ff; border-radius:6px; padding:10px 14px; color:#58a6ff; font-size:14px; }
    [data-testid="stSidebar"] { background-color:#161b22; border-right:1px solid #30363d; }
    div[data-testid="metric-container"] { display:none; }
    .stPlotlyChart { border-radius:12px; overflow:hidden; }
    .fbadge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin-left:6px; }
    .f-real { background:#1b2d1b; color:#3fb950; border:1px solid #3fb950; }
    .f-sim  { background:#1c2128; color:#7d8590; border:1px solid #30363d; }
    .f-extra { background:#1b1f2d; color:#58a6ff; border:1px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "bg":"#0d1117","surface":"#161b22","border":"#30363d","text":"#e6edf3",
    "muted":"#7d8590","blue":"#58a6ff","orange":"#f0883e","green":"#3fb950","red":"#f85149",
}

PALETA_EXTRA = ["#f0883e","#bc8cff","#ffa657","#79c0ff","#56d364","#ff7b72"]


# ─────────────────────────────────────────────
# LECTORES DE EXCEL
# ─────────────────────────────────────────────

def leer_horario_xm(archivo, hoja, nombre_col, factor=1.0):
    """Excel formato XM: Fecha | 0..23. Promedia 24h, aplica factor."""
    ruta = os.path.join(DIR, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, header=0)
        df = df.rename(columns={df.columns[0]: "fecha"})
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])
        cols_h = [c for c in df.columns if str(c).strip() in [str(h) for h in range(24)]]
        df[nombre_col] = df[cols_h].apply(pd.to_numeric, errors="coerce").mean(axis=1) * factor
        df = df[["fecha", nombre_col]].dropna()
        return df.sort_values("fecha").reset_index(drop=True), True
    except Exception as e:
        st.sidebar.warning(f"⚠ {archivo}: {e}")
        return None, False


def leer_diario(archivo, hoja, col_fecha, col_valor, nombre_col):
    """Excel con formato diario simple: col_fecha | col_valor."""
    ruta = os.path.join(DIR, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, header=0)
        df = df.rename(columns={col_fecha: "fecha", col_valor: nombre_col})
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df[nombre_col] = pd.to_numeric(df[nombre_col], errors="coerce")
        df = df[["fecha", nombre_col]].dropna()
        return df.sort_values("fecha").reset_index(drop=True), True
    except Exception as e:
        st.sidebar.warning(f"⚠ {archivo}: {e}")
        return None, False


def leer_variable_extra(archivo, nombre_col):
    """
    Lee un Excel de variable extra con exactamente 2 columnas: fecha | valor.
    Acepta cualquier nombre de columna — usa posición.
    Interpola gaps y hace ffill/bfill.
    """
    ruta = os.path.join(DIR, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, header=0)
        # Tomar las dos primeras columnas útiles
        df = df.iloc[:, :2].copy()
        df.columns = ["fecha", nombre_col]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df[nombre_col] = pd.to_numeric(df[nombre_col], errors="coerce")
        df = df.dropna(subset=["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)
        # Expandir a diario completo con interpolación
        fechas = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
        df_d = pd.DataFrame({"fecha": fechas})
        df_d = df_d.merge(df, on="fecha", how="left")
        df_d[nombre_col] = df_d[nombre_col].interpolate().ffill().bfill()
        return df_d, True
    except Exception as e:
        st.sidebar.warning(f"⚠ {archivo}: {e}")
        return None, False


def leer_oni(archivo, hoja):
    """ONI mensual → serie diaria por ffill."""
    ruta = os.path.join(DIR, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, header=0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["oni"]   = pd.to_numeric(df["oni"], errors="coerce")
        df = df[["fecha", "oni"]].dropna().sort_values("fecha")
        fechas = pd.date_range(df["fecha"].min(), date.today(), freq="D")
        df_d = pd.DataFrame({"fecha": fechas})
        df_d = df_d.merge(df, on="fecha", how="left")
        df_d["oni"] = df_d["oni"].ffill().bfill().fillna(0.0)
        return df_d.set_index("fecha")["oni"], True
    except Exception as e:
        st.sidebar.warning(f"⚠ {archivo}: {e}")
        return None, False


# ─────────────────────────────────────────────
# CARGA Y ENSAMBLE
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cargar_datos():
    fuentes = {}
    vars_extra_cargadas = {}  # {nombre_col: unidad}

    # ── Precio (obligatorio) ──
    df_precio, ok = leer_horario_xm("PrecioBolsa2026.xlsx", "PrecioBolsa", "precio", factor=1.0)
    if not ok or df_precio is None:
        st.error("❌ No se encontró PrecioBolsa2026.xlsx — archivo obligatorio.")
        st.stop()
    df = df_precio.copy()
    fuentes["precio"] = "real"

    n = len(df)
    t = np.arange(n)
    np.random.seed(42)
    est = 30 * np.sin(2 * np.pi * t / 365 - np.pi / 2)

    # ── Aportes ──
    df_ap, ok_ap = leer_horario_xm("Aportes2026.xlsx", "Aportes", "aportes", factor=1e-6)
    if ok_ap and df_ap is not None:
        df = df.merge(df_ap, on="fecha", how="left")
        df["aportes"] = df["aportes"].interpolate().ffill().bfill()
        fuentes["aportes"] = "real"
    else:
        df["aportes"] = np.round(np.clip(3500 - 20*est + np.random.normal(0,200,n), 500, 7000), 1)
        fuentes["aportes"] = "simulado"

    # ── Embalses ──
    df_em, ok_em = leer_diario(
        "Embalses2026.xlsx", "Reservas_Diario_SIN",
        "Fecha", "Volumen Útil Diario %", "embalses"
    )
    if ok_em and df_em is not None:
        df = df.merge(df_em, on="fecha", how="left")
        df["embalses"] = df["embalses"].interpolate().ffill().bfill()
        fuentes["embalses"] = "real"
    else:
        df["embalses"] = np.round(np.clip(65 - 0.3*est + np.random.normal(0,5,n), 10, 100), 1)
        fuentes["embalses"] = "simulado"

    # ── Demanda ──
    df_dm, ok_dm = leer_horario_xm("Demanda2026.xlsx", "Demanda", "demanda", factor=1e-6)
    if ok_dm and df_dm is not None:
        df = df.merge(df_dm, on="fecha", how="left")
        df["demanda"] = df["demanda"].interpolate().ffill().bfill()
        fuentes["demanda"] = "real"
    else:
        df["demanda"] = np.round(165 + 0.005*t + 5*np.sin(2*np.pi*t/365) + np.random.normal(0,3,n), 1)
        fuentes["demanda"] = "simulado"

    # ── ONI ──
    oni_serie, ok_oni = leer_oni("ONI.xlsx", "ONI")
    if ok_oni and oni_serie is not None:
        df["oni"] = df["fecha"].map(oni_serie).ffill().bfill().fillna(0.0)
        fuentes["oni"] = "real"
    else:
        df["oni"] = 0.0
        fuentes["oni"] = "simulado"

    # ── Variables extra (automático) ──
    for nombre_col, (archivo, prior_scale, unidad) in VARIABLES_EXTRA.items():
        df_extra, ok_extra = leer_variable_extra(archivo, nombre_col)
        if ok_extra and df_extra is not None and len(df_extra) > 0:
            df = df.merge(df_extra, on="fecha", how="left")
            df[nombre_col] = df[nombre_col].interpolate().ffill().bfill()
            fuentes[nombre_col] = "real"
            vars_extra_cargadas[nombre_col] = unidad
        # Si no existe el archivo, simplemente no se agrega — no se simula

    # Deduplicar fechas
    df = df.groupby("fecha", as_index=False).mean(numeric_only=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    return df, fuentes, vars_extra_cargadas


@st.cache_data(ttl=3600)
def entrenar_y_pronosticar(horizonte_dias: int):
    df, fuentes, vars_extra_cargadas = cargar_datos()

    dp = df.rename(columns={"fecha": "ds", "precio": "y"}).copy()

    # Columnas base + extra disponibles
    cols_base  = ["aportes", "embalses", "demanda", "oni"]
    cols_extra = list(vars_extra_cargadas.keys())
    todas_cols = cols_base + [c for c in cols_extra if c in dp.columns]

    prior_scales_base = {
        "aportes_norm": 0.5, "embalses_norm": 0.4,
        "demanda_norm": 0.3, "oni_norm": 0.6,
    }
    prior_scales_extra = {
        f"{c}_norm": VARIABLES_EXTRA[c][1]
        for c in cols_extra if c in VARIABLES_EXTRA
    }
    prior_scales = {**prior_scales_base, **prior_scales_extra}

    regresores = []
    for col in todas_cols:
        if col not in dp.columns:
            continue
        mu  = dp[col].mean()
        std = dp[col].std()
        if std == 0 or np.isnan(std): std = 1.0
        dp[f"{col}_norm"] = (dp[col] - mu) / std
        if dp[f"{col}_norm"].isna().sum() == 0:
            regresores.append(f"{col}_norm")

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    for reg in regresores:
        m.add_regressor(reg, prior_scale=prior_scales.get(reg, 0.4))

    m.fit(dp)

    futuro = m.make_future_dataframe(periods=horizonte_dias, freq="D")
    ultima = dp["ds"].max()

    for col in regresores:
        hist_rec = dp[dp["ds"] >= ultima - pd.Timedelta(days=90)][col]
        mu_r  = hist_rec.mean()
        std_r = max(hist_rec.std() * 0.3, 0.01)
        proy  = np.random.normal(mu_r, std_r, horizonte_dias)
        mapa  = dp.drop_duplicates("ds").set_index("ds")[col]
        vals  = futuro["ds"].map(mapa).values.astype(float)
        vals  = np.where(np.isnan(vals), mu_r, vals)
        mask_futuro = futuro["ds"] > ultima
        vals[mask_futuro.values] = proy[:mask_futuro.sum()]
        futuro[col] = vals

    forecast = m.predict(futuro)
    return df, dp, forecast, m, fuentes, regresores, vars_extra_cargadas


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Configuración")
    st.markdown("---")
    horizonte       = st.slider("Horizonte de pronóstico (días)", 7, 90, 30, step=7)
    umbral_alto_pct = st.slider("Percentil alerta alta", 70, 95, 85, step=5)
    umbral_bajo_pct = st.slider("Percentil alerta baja", 5, 30, 15, step=5)
    mostrar_comp    = st.checkbox("Mostrar componentes del modelo", value=False)
    mostrar_tabla   = st.checkbox("Mostrar tabla de pronóstico", value=True)
    st.markdown("---")
    st.markdown("""
**Archivos base:**
- `PrecioBolsa2026.xlsx` ✅
- `Aportes2026.xlsx`
- `Embalses2026.xlsx`
- `Demanda2026.xlsx`
- `ONI.xlsx`

**Variables extra (opcionales):**
- `WTI.xlsx` → precio petróleo
- `USDCOP.xlsx` → tasa de cambio
- `Gas.xlsx` → precio gas
- `Carbon.xlsx` → precio carbón

*Formato: 2 columnas → fecha | valor*
    """)
    st.markdown("---")
    st.markdown("*Prophet · XM + NOAA*")


# ─────────────────────────────────────────────
# CABECERA
# ─────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:700;color:#e6edf3;margin-bottom:0;'>
    ⚡ Precio de Bolsa · Colombia
</h1>
<p style='color:#7d8590;font-size:14px;margin-top:4px;'>
    Pronóstico · Prophet · XM + NOAA 2026
</p>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

with st.spinner("Cargando datos y entrenando modelo..."):
    df, dp, forecast, modelo, fuentes, regresores, vars_extra_cargadas = entrenar_y_pronosticar(horizonte)

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


# ─────────────────────────────────────────────
# BADGES FUENTE
# ─────────────────────────────────────────────

def badge(var):
    estado = fuentes.get(var, "no")
    if estado == "real":
        return f'<span class="fbadge f-real">✓ real</span>'
    elif estado == "simulado":
        return f'<span class="fbadge f-sim">≈ sim</span>'
    else:
        return f'<span class="fbadge f-sim">— n/d</span>'

badges_base = (
    f"Precio {badge('precio')} &nbsp;"
    f"Aportes {badge('aportes')} &nbsp;"
    f"Embalses {badge('embalses')} &nbsp;"
    f"Demanda {badge('demanda')} &nbsp;"
    f"ONI {badge('oni')}"
)
badges_extra = " &nbsp;".join(
    f"{col.upper()} {badge(col)}" for col in vars_extra_cargadas
)
st.markdown(
    badges_base + (" &nbsp;&nbsp;|&nbsp;&nbsp; " + badges_extra if badges_extra else ""),
    unsafe_allow_html=True
)
if vars_extra_cargadas:
    st.markdown(
        f'<span style="color:#58a6ff;font-size:12px;">⚡ Variables extra activas en el modelo: '
        f'{", ".join(v.upper() for v in vars_extra_cargadas)}</span>',
        unsafe_allow_html=True
    )
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

def kpi(label, value, delta=None, delta_label=""):
    dh = ""
    if delta is not None:
        cls  = "kpi-delta-up" if delta > 0 else ("kpi-delta-down" if delta < 0 else "kpi-delta-flat")
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        dh = f'<div class="{cls}">{sign} {abs(delta):.1f}% {delta_label}</div>'
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{dh}</div>'

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(kpi("Precio actual",               f"${precio_hoy:.0f}"),                                    unsafe_allow_html=True)
with c2: st.markdown(kpi("Pronóstico mañana",           f"${precio_fc_d1:.0f}", delta=delta_pct, delta_label="vs hoy"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"Pronóstico día {horizonte}", f"${precio_fc_fn:.0f}"),                                   unsafe_allow_html=True)
with c4: st.markdown(kpi("Nivel embalses",              f"{df['embalses'].iloc[-1]:.1f}%"),                        unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ALERTA
# ─────────────────────────────────────────────

alertas_map = {
    "ALTO":   ("alerta-alto",   f"⚠ Alerta: precio pronosticado ALTO (>${umbral_alto:.0f}/kWh · p{umbral_alto_pct})"),
    "BAJO":   ("alerta-bajo",   f"✓ Precio pronosticado BAJO (<${umbral_bajo:.0f}/kWh · p{umbral_bajo_pct}) — oportunidad"),
    "NORMAL": ("alerta-normal", f"● Precio pronosticado en rango NORMAL"),
}
css_cls, msg = alertas_map[alerta_nivel]
st.markdown(f'<div class="{css_cls}">{msg}</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tabs_labels = ["📈 Pronóstico", "🔍 Variables base"]
if vars_extra_cargadas:
    tabs_labels.append("📊 Variables extra")

tabs = st.tabs(tabs_labels)

# ── Tab 1: Pronóstico ──
with tabs[0]:
    n_dias   = len(dp)
    opciones = [x for x in [30, 60, 90, 150, 365, 730] if x <= n_dias] or [n_dias]
    dias_hist = st.select_slider(
        "Histórico a mostrar",
        options=opciones,
        value=opciones[-1],
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
        mode="lines", line=dict(color=COLORS["muted"], width=1.2), name="Real"))
    fig.add_trace(go.Scatter(x=fh["ds"], y=fh["yhat"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5, dash="dot"),
        name="Ajuste modelo", opacity=0.8))
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
    st.caption(f"Regresores activos: {', '.join(regresores)}")

    if mostrar_comp:
        st.markdown("**Componentes del modelo**")
        st.pyplot(modelo.plot_components(forecast))


# ── Tab 2: Variables base ──
with tabs[1]:
    df_plot = df.rename(columns={"fecha": "ds"})

    def mini_line(x, y, color, title, ylabel):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=1.5), fill="tozeroy",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
        ))
        f.update_layout(
            title=dict(text=title, font=dict(size=13, color=COLORS["text"])),
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
            font=dict(family="DM Sans", color=COLORS["text"]),
            yaxis=dict(title=ylabel, gridcolor=COLORS["border"], zeroline=False),
            xaxis=dict(gridcolor=COLORS["border"]),
            height=250, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        )
        return f

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(mini_line(df_plot["ds"], df_plot["aportes"], "#58a6ff",
            f"Aportes hídricos {'✓' if fuentes['aportes']=='real' else '≈'}", "GWh/día"),
            use_container_width=True)
        st.plotly_chart(mini_line(df_plot["ds"], df_plot["demanda"], "#bc8cff",
            f"Demanda nacional {'✓' if fuentes['demanda']=='real' else '≈'}", "GWh/día"),
            use_container_width=True)
    with col_b:
        st.plotly_chart(mini_line(df_plot["ds"], df_plot["embalses"], "#3fb950",
            f"Nivel embalses {'✓' if fuentes['embalses']=='real' else '≈'}", "%"),
            use_container_width=True)
        if fuentes["oni"] == "real":
            fig_oni = go.Figure()
            fig_oni.add_trace(go.Bar(
                x=df_plot["ds"], y=df_plot["oni"],
                marker_color=["#f85149" if v > 0 else "#58a6ff" for v in df_plot["oni"]],
            ))
            fig_oni.update_layout(
                title=dict(text="Índice ONI/ENSO ✓", font=dict(size=13, color=COLORS["text"])),
                paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["surface"],
                font=dict(family="DM Sans", color=COLORS["text"]),
                yaxis=dict(title="°C anomalía", gridcolor=COLORS["border"],
                           zeroline=True, zerolinecolor=COLORS["border"]),
                xaxis=dict(gridcolor=COLORS["border"]),
                height=250, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
            )
            st.plotly_chart(fig_oni, use_container_width=True)
        else:
            st.markdown(
                '<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                'padding:20px;text-align:center;color:#7d8590;font-size:13px;height:250px;'
                'display:flex;align-items:center;justify-content:center;">'
                '📡 ONI/ENSO · Sube ONI.xlsx para activar</div>',
                unsafe_allow_html=True
            )


# ── Tab 3: Variables extra (solo si hay alguna cargada) ──
if vars_extra_cargadas and len(tabs) > 2:
    with tabs[2]:
        st.markdown("Variables externas activas como regresores del modelo.")
        items = list(vars_extra_cargadas.items())
        # Grid de 2 columnas
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, (nombre_col, unidad) in enumerate(items[i:i+2]):
                if nombre_col not in df_plot.columns:
                    continue
                color = PALETA_EXTRA[i + j % len(PALETA_EXTRA)]
                with cols[j]:
                    st.plotly_chart(
                        mini_line(df_plot["ds"], df_plot[nombre_col], color,
                                  f"{nombre_col.upper()} ✓", unidad),
                        use_container_width=True
                    )


# ─────────────────────────────────────────────
# TABLA + DESCARGA
# ─────────────────────────────────────────────

if mostrar_tabla:
    st.markdown("### Tabla de pronóstico")
    tabla = futuro_fc[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    tabla.columns = ["Fecha","Pronóstico ($/kWh)","Límite inferior 80%","Límite superior 80%"]
    tabla["Fecha"] = tabla["Fecha"].dt.strftime("%Y-%m-%d")
    tabla["Alerta"] = tabla["Pronóstico ($/kWh)"].apply(
        lambda p: "🔴 ALTO" if p > umbral_alto else ("🟢 BAJO" if p < umbral_bajo else "🔵 NORMAL")
    )
    for col in ["Pronóstico ($/kWh)","Límite inferior 80%","Límite superior 80%"]:
        tabla[col] = tabla[col].round(1)
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar CSV", data=csv,
        file_name=f"pronostico_energia_{date.today()}.csv", mime="text/csv")


# ─────────────────────────────────────────────
# MÉTRICAS
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
        "Métricas in-sample — el modelo entrenó con estos datos. "
        "Con más meses históricos la precisión out-of-sample mejora significativamente."
    )
