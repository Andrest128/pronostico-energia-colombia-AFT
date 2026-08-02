"""
price_forecast.py
───────────────────
DECISIÓN DE DISEÑO IMPORTANTE
------------------------------
El archivo `app.py` del dashboard mezcla la lógica de negocio (leer
Excel, entrenar Prophet) con llamadas a Streamlit (`st.cache_data`,
`st.error`, `st.sidebar.warning`). Eso es válido para una app de una
sola página, pero significa que esas funciones NO se pueden importar
directamente en un script que no corra dentro de `streamlit run`
(como este optimizador, o un backtest, o un notebook).

Este módulo replica exactamente la misma lógica de lectura y
entrenamiento (mismos parámetros de Prophet, mismos regresores, mismo
manejo de fuentes reales/simuladas) pero sin ninguna dependencia de
Streamlit, para que:
  1. El optimizador pueda generar pronósticos consistentes con el
     dashboard, sin duplicar lógica "a mano" y sin riesgo de que las
     dos versiones diverjan con el tiempo.
  2. `app.py` pueda, si se quiere más adelante, importar estas mismas
     funciones y quedar como una capa delgada de UI sobre esta lógica
     (recomendado — ver README.md, sección "Refactor sugerido").

Por ahora este archivo es independiente de app.py (no lo modifica) para
no arriesgar el dashboard en producción. La duplicación es intencional
y temporal — está documentada aquí para que no se te olvide.
"""

import os
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet

warnings.filterwarnings("ignore")

# Debe coincidir exactamente con VARIABLES_EXTRA en app.py.
# Si agregas una variable extra en el dashboard, replícala aquí.
VARIABLES_EXTRA = {
    "wti":    ("WTI.xlsx",    0.4, "USD/barril"),
    "usdcop": ("USDCOP.xlsx", 0.4, "COP/USD"),
    "gas":    ("Gas.xlsx",    0.4, "USD/MMBTU"),
    "carbon": ("Carbon.xlsx", 0.3, "USD/ton"),
}


def _warn(msg: str):
    print(f"[price_forecast] ADVERTENCIA: {msg}")


def leer_horario_xm(data_dir, archivo, hoja, nombre_col, factor=1.0):
    ruta = os.path.join(data_dir, archivo)
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
        _warn(f"{archivo}: {e}")
        return None, False


def leer_diario(data_dir, archivo, hoja, col_fecha, col_valor, nombre_col):
    ruta = os.path.join(data_dir, archivo)
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
        _warn(f"{archivo}: {e}")
        return None, False


def leer_variable_extra(data_dir, archivo, nombre_col):
    ruta = os.path.join(data_dir, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, header=0)
        df = df.iloc[:, :2].copy()
        df.columns = ["fecha", nombre_col]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df[nombre_col] = pd.to_numeric(df[nombre_col], errors="coerce")
        df = df.dropna(subset=["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)
        fechas = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
        df_d = pd.DataFrame({"fecha": fechas})
        df_d = df_d.merge(df, on="fecha", how="left")
        df_d[nombre_col] = df_d[nombre_col].interpolate().ffill().bfill()
        return df_d, True
    except Exception as e:
        _warn(f"{archivo}: {e}")
        return None, False


def leer_oni(data_dir, archivo, hoja):
    from datetime import date
    ruta = os.path.join(data_dir, archivo)
    if not os.path.exists(ruta):
        return None, False
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, header=0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["oni"] = pd.to_numeric(df["oni"], errors="coerce")
        df = df[["fecha", "oni"]].dropna().sort_values("fecha")
        fechas = pd.date_range(df["fecha"].min(), date.today(), freq="D")
        df_d = pd.DataFrame({"fecha": fechas})
        df_d = df_d.merge(df, on="fecha", how="left")
        df_d["oni"] = df_d["oni"].ffill().bfill().fillna(0.0)
        return df_d.set_index("fecha")["oni"], True
    except Exception as e:
        _warn(f"{archivo}: {e}")
        return None, False


def cargar_datos(data_dir: str):
    """Idéntico en lógica a `cargar_datos()` de app.py, sin cache de Streamlit."""
    fuentes = {}
    vars_extra_cargadas = {}

    df_precio, ok = leer_horario_xm(data_dir, "PrecioBolsa2026.xlsx", "PrecioBolsa", "precio", factor=1.0)
    if not ok or df_precio is None:
        raise FileNotFoundError("No se encontró PrecioBolsa2026.xlsx — archivo obligatorio.")
    df = df_precio.copy()
    fuentes["precio"] = "real"

    n = len(df)
    t = np.arange(n)
    np.random.seed(42)
    est = 30 * np.sin(2 * np.pi * t / 365 - np.pi / 2)

    df_ap, ok_ap = leer_horario_xm(data_dir, "Aportes2026.xlsx", "Aportes", "aportes", factor=1e-6)
    if ok_ap and df_ap is not None:
        df = df.merge(df_ap, on="fecha", how="left")
        df["aportes"] = df["aportes"].interpolate().ffill().bfill()
        fuentes["aportes"] = "real"
    else:
        df["aportes"] = np.round(np.clip(3500 - 20 * est + np.random.normal(0, 200, n), 500, 7000), 1)
        fuentes["aportes"] = "simulado"

    df_em, ok_em = leer_diario(data_dir, "Embalses2026.xlsx", "Reservas_Diario_SIN", "Fecha", "Volumen Útil Diario %", "embalses")
    if ok_em and df_em is not None:
        df = df.merge(df_em, on="fecha", how="left")
        df["embalses"] = df["embalses"].interpolate().ffill().bfill()
        fuentes["embalses"] = "real"
    else:
        df["embalses"] = np.round(np.clip(65 - 0.3 * est + np.random.normal(0, 5, n), 10, 100), 1)
        fuentes["embalses"] = "simulado"

    df_dm, ok_dm = leer_horario_xm(data_dir, "Demanda2026.xlsx", "Demanda", "demanda", factor=1e-6)
    if ok_dm and df_dm is not None:
        df = df.merge(df_dm, on="fecha", how="left")
        df["demanda"] = df["demanda"].interpolate().ffill().bfill()
        fuentes["demanda"] = "real"
    else:
        df["demanda"] = np.round(165 + 0.005 * t + 5 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, n), 1)
        fuentes["demanda"] = "simulado"

    oni_serie, ok_oni = leer_oni(data_dir, "ONI.xlsx", "ONI")
    if ok_oni and oni_serie is not None:
        df["oni"] = df["fecha"].map(oni_serie).ffill().bfill().fillna(0.0)
        fuentes["oni"] = "real"
    else:
        df["oni"] = 0.0
        fuentes["oni"] = "simulado"

    for nombre_col, (archivo, prior_scale, unidad) in VARIABLES_EXTRA.items():
        df_extra, ok_extra = leer_variable_extra(data_dir, archivo, nombre_col)
        if ok_extra and df_extra is not None and len(df_extra) > 0:
            df = df.merge(df_extra, on="fecha", how="left")
            df[nombre_col] = df[nombre_col].interpolate().ffill().bfill()
            fuentes[nombre_col] = "real"
            vars_extra_cargadas[nombre_col] = unidad

    df = df.groupby("fecha", as_index=False).mean(numeric_only=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    return df, fuentes, vars_extra_cargadas


def entrenar_y_pronosticar(data_dir: str, horizonte_dias: int):
    """Idéntico en lógica a `entrenar_y_pronosticar()` de app.py, sin cache de Streamlit."""
    df, fuentes, vars_extra_cargadas = cargar_datos(data_dir)
    dp = df.rename(columns={"fecha": "ds", "precio": "y"}).copy()

    cols_base = ["aportes", "embalses", "demanda", "oni"]
    cols_extra = list(vars_extra_cargadas.keys())
    todas_cols = cols_base + [c for c in cols_extra if c in dp.columns]

    prior_scales_base = {"aportes_norm": 0.5, "embalses_norm": 0.4, "demanda_norm": 0.3, "oni_norm": 0.6}
    prior_scales_extra = {f"{c}_norm": VARIABLES_EXTRA[c][1] for c in cols_extra if c in VARIABLES_EXTRA}
    prior_scales = {**prior_scales_base, **prior_scales_extra}

    regresores = []
    for col in todas_cols:
        if col not in dp.columns:
            continue
        mu, std = dp[col].mean(), dp[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        dp[f"{col}_norm"] = (dp[col] - mu) / std
        if dp[f"{col}_norm"].isna().sum() == 0:
            regresores.append(f"{col}_norm")

    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        interval_width=0.80, changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0, seasonality_mode="multiplicative",
    )
    for reg in regresores:
        m.add_regressor(reg, prior_scale=prior_scales.get(reg, 0.4))
    m.fit(dp)

    futuro = m.make_future_dataframe(periods=horizonte_dias, freq="D")
    ultima = dp["ds"].max()
    for col in regresores:
        hist_rec = dp[dp["ds"] >= ultima - pd.Timedelta(days=90)][col]
        mu_r = hist_rec.mean()
        std_r = max(hist_rec.std() * 0.3, 0.01)
        proy = np.random.normal(mu_r, std_r, horizonte_dias)
        mapa = dp.drop_duplicates("ds").set_index("ds")[col]
        vals = futuro["ds"].map(mapa).values.astype(float)
        vals = np.where(np.isnan(vals), mu_r, vals)
        mask_futuro = futuro["ds"] > ultima
        vals[mask_futuro.values] = proy[: mask_futuro.sum()]
        futuro[col] = vals

    forecast = m.predict(futuro)
    return df, dp, forecast, m, fuentes, regresores, vars_extra_cargadas
