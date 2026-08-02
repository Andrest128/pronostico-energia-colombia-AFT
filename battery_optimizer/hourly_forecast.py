"""
hourly_forecast.py
────────────────────
Combina dos piezas que ya existen por separado:
  1. El pronóstico DIARIO de Prophet (price_forecast.entrenar_y_pronosticar)
  2. La forma horaria histórica (hourly_shape.calcular_forma_horaria)

...para reconstruir un pronóstico HORARIO, que es lo que el optimizador
de despacho de la batería necesita (no puede optimizar arbitraje con un
solo precio promedio del día).

Fórmula: precio_hora(d, h) = precio_pronosticado_dia(d) * forma_horaria(h)

Esto preserva exactamente el pronóstico diario que ya validaste en el
dashboard (mismo nivel, mismas alertas altas/bajas) y solo le agrega
la variación horaria dentro de cada día.
"""

import pandas as pd
import numpy as np


def construir_pronostico_horario(
    forecast_diario: pd.DataFrame,
    forma_horaria,
    fecha_inicio,
    horizonte_dias: int,
) -> pd.DataFrame:
    """
    Parámetros
    ----------
    forecast_diario : DataFrame de salida de Prophet (`forecast` en price_forecast.py),
        con columnas ds, yhat, yhat_lower, yhat_upper.
    forma_horaria : Series (index 0..23) o dict {"semana": Series, "finde": Series},
        salida de hourly_shape.calcular_forma_horaria.
    fecha_inicio : primer día del horizonte a expandir (normalmente "mañana",
        el primer día después del último dato histórico).
    horizonte_dias : cuántos días expandir a horario.

    Retorna
    -------
    DataFrame con columnas: fecha, hora, datetime, precio_yhat, precio_lower, precio_upper
    """
    dias = pd.date_range(fecha_inicio, periods=horizonte_dias, freq="D")
    fc = forecast_diario.set_index("ds")

    filas = []
    for d in dias:
        if d not in fc.index:
            continue
        yhat = fc.loc[d, "yhat"]
        lower = fc.loc[d, "yhat_lower"]
        upper = fc.loc[d, "yhat_upper"]

        if isinstance(forma_horaria, dict):
            shape = forma_horaria["finde"] if d.dayofweek >= 5 else forma_horaria["semana"]
        else:
            shape = forma_horaria

        for h in range(24):
            mult = shape.loc[h]
            filas.append({
                "fecha": d.date(),
                "hora": h,
                "datetime": d + pd.Timedelta(hours=h),
                "precio_yhat": max(yhat * mult, 0.0),
                "precio_lower": max(lower * mult, 0.0),
                "precio_upper": max(upper * mult, 0.0),
            })

    return pd.DataFrame(filas)
