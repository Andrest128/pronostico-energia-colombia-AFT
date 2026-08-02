"""
hourly_shape.py
────────────────
PROBLEMA QUE RESUELVE ESTE MÓDULO
----------------------------------
El modelo de pronóstico existente (app.py) entrena Prophet sobre el
PROMEDIO DIARIO del precio de bolsa (colapsa las 24 columnas horarias
del Excel en un solo número por día). Esto es correcto para el caso de
uso original del dashboard (ver tendencia y alertas de precio), pero
para optimizar el despacho de una batería ese promedio diario no sirve:
el valor de un BESS viene precisamente de la diferencia de precio
DENTRO del día (comprar barato de madrugada, vender caro en la punta
de la noche).

Con los datos reales del repo, el spread intradía (máximo-mínimo del
día) representa en promedio ~119% del precio promedio diario de los
últimos 90 días — es una señal demasiado grande para ignorar.

SOLUCIÓN
--------
En vez de re-entrenar Prophet hora por hora (24 modelos, mucho más
costoso y con menos datos por hora), este módulo calcula una "forma
horaria" a partir del histórico real: para cada hora del día, el
promedio de (precio_hora / precio_promedio_del_día) en una ventana
reciente. Esa forma se multiplica luego por el pronóstico diario de
Prophet (ver hourly_forecast.py) para reconstruir un pronóstico horario
razonable, sin tocar ni reentrenar el modelo Prophet existente.

Limitación conocida (documentada a propósito): esta forma horaria es
un patrón PROMEDIO histórico, no un pronóstico hora-por-hora
propiamente dicho. Asume que la forma del día (cuándo son las horas
caras/baratas) se mantiene razonablemente estable en el horizonte de
pronóstico. Es una aproximación estándar en la industria (conocida como
"shape-based disaggregation") y muy superior a asumir precio plano,
pero si la forma del despacho cambia estructuralmente (ej. entra mucha
solar nueva y desplaza el pico), hay que re-calcular la forma con datos
más recientes o segmentarla por mes/estación.
"""

import os
import pandas as pd
import numpy as np

HORAS = [str(h) for h in range(24)]


def cargar_precio_horario_crudo(ruta_archivo: str, hoja: str = "PrecioBolsa") -> pd.DataFrame:
    """
    Lee el Excel de precio de bolsa SIN colapsar las horas (a diferencia
    de `leer_horario_xm` en app.py, que promedia). Devuelve un DataFrame
    con columnas: fecha, 0, 1, ..., 23 (todas numéricas), ordenado por fecha.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    df = pd.read_excel(ruta_archivo, sheet_name=hoja, header=0)
    df = df.rename(columns={df.columns[0]: "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    for h in HORAS:
        if h not in df.columns:
            raise ValueError(f"Falta la columna de hora '{h}' en {ruta_archivo}")
    df[HORAS] = df[HORAS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=HORAS, how="all")
    return df[["fecha"] + HORAS].sort_values("fecha").reset_index(drop=True)


def calcular_forma_horaria(
    df_horario: pd.DataFrame,
    dias_recientes: int = 180,
    por_tipo_dia: bool = False,
) -> pd.DataFrame:
    """
    Calcula el multiplicador horario normalizado (precio_hora / precio_promedio_dia),
    promediado sobre los últimos `dias_recientes` días con datos.

    Parámetros
    ----------
    df_horario : DataFrame de `cargar_precio_horario_crudo`
    dias_recientes : ventana de días hacia atrás usada para calcular la forma.
        180 días (~6 meses) balancea estabilidad con capacidad de capturar
        cambios recientes (ej. nueva entrada solar, cambios regulatorios).
    por_tipo_dia : si True, calcula una forma separada para días de semana
        (lunes-viernes) y fines de semana/festivos aproximados (sábado-domingo).
        La demanda y el precio suelen tener perfiles distintos entre semana
        y fin de semana en Colombia.

    Retorna
    -------
    Si por_tipo_dia=False: Series indexada 0..23 con el multiplicador promedio.
    Si por_tipo_dia=True: dict {"semana": Series, "finde": Series}.
    """
    recientes = df_horario.tail(dias_recientes).copy()
    if len(recientes) == 0:
        raise ValueError("No hay datos suficientes para calcular la forma horaria.")

    def _shape(sub: pd.DataFrame) -> pd.Series:
        promedio_dia = sub[HORAS].mean(axis=1)
        promedio_dia = promedio_dia.replace(0, np.nan)
        norm = sub[HORAS].div(promedio_dia, axis=0)
        shape = norm.mean(axis=0)
        shape.index = shape.index.astype(int)
        return shape.sort_index()

    if not por_tipo_dia:
        return _shape(recientes)

    recientes["dow"] = recientes["fecha"].dt.dayofweek
    semana = recientes[recientes["dow"] < 5]
    finde = recientes[recientes["dow"] >= 5]
    return {
        "semana": _shape(semana) if len(semana) > 0 else _shape(recientes),
        "finde": _shape(finde) if len(finde) > 0 else _shape(recientes),
    }


def resumen_spread_intradia(df_horario: pd.DataFrame, dias_recientes: int = 90) -> dict:
    """
    Calcula métricas simples del spread intradía reciente — útil para
    justificar (con números reales) por qué vale la pena optimizar hora a
    hora y no solo con el promedio diario.
    """
    recientes = df_horario.tail(dias_recientes).copy()
    promedio_dia = recientes[HORAS].mean(axis=1)
    min_dia = recientes[HORAS].min(axis=1)
    max_dia = recientes[HORAS].max(axis=1)
    spread = max_dia - min_dia
    return {
        "dias_analizados": len(recientes),
        "precio_promedio_dia": round(float(promedio_dia.mean()), 1),
        "spread_promedio": round(float(spread.mean()), 1),
        "spread_pct_del_promedio": round(float((spread / promedio_dia).mean() * 100), 1),
    }
