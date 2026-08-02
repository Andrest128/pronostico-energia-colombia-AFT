"""
backtest.py
─────────────
Corre el optimizador (LP) y la estrategia naive sobre una ventana de
DÍAS REALES CONSECUTIVOS del histórico de precio de bolsa, encadenando
el estado de carga (SOC) de un día al siguiente — a diferencia de la
prueba de un solo día, esto expone diferencias que solo aparecen con
más de un ciclo de por medio (ej. dejar la batería a medio cargar
porque mañana el precio va a estar mejor, algo que un heurístico de
"un ciclo por día" no puede ver).

Este backtest usa PRECIOS REALES YA OCURRIDOS (no el pronóstico) — es
decir, mide el techo teórico de ingresos si se hubiera conocido el
precio exacto de antemano ("perfect foresight"). Es una medida útil
para dimensionar el valor máximo posible de la batería, pero en
producción el optimizador correrá contra el PRONÓSTICO (con su error
correspondiente) — ver demo_pronostico_futuro.py para esa versión.
"""

import sys
sys.path.insert(0, ".")
import pandas as pd

from battery_model import BatteryConfig
from optimizer import optimizar_despacho, resumen_resultado
from naive_strategy import despacho_naive


def backtest_multi_dia(
    df_horario: pd.DataFrame,
    battery: BatteryConfig,
    dias: int = 30,
) -> pd.DataFrame:
    """
    Parámetros
    ----------
    df_horario : DataFrame con columnas fecha, 0..23 (salida de
        hourly_shape.cargar_precio_horario_crudo)
    battery : configuración de la batería
    dias : cuántos días recientes incluir en el backtest

    Retorna
    -------
    DataFrame con una fila por día: fecha, ingreso_lp, ingreso_naive,
    mejora_pct, soc_final_lp, soc_final_naive
    """
    HORAS = [str(h) for h in range(24)]
    ventana = df_horario.tail(dias).reset_index(drop=True)

    soc_lp = battery.soc_init_mwh
    soc_naive = battery.soc_init_mwh
    filas = []

    for _, fila in ventana.iterrows():
        precios_dia = pd.Series(fila[HORAS].values.astype(float))

        res_lp = optimizar_despacho(precios_dia, battery, soc_inicial_mwh=soc_lp)
        res_naive = despacho_naive(precios_dia, battery, soc_inicial_mwh=soc_naive)

        resumen_lp = resumen_resultado(res_lp)
        resumen_naive = resumen_resultado(res_naive)

        soc_lp = res_lp["soc_mwh"].iloc[-1]
        soc_naive = res_naive["soc_mwh"].iloc[-1]

        filas.append({
            "fecha": fila["fecha"].date(),
            "ingreso_lp": resumen_lp["ingreso_total"],
            "ingreso_naive": resumen_naive["ingreso_total"],
            "soc_final_lp_pct": round(soc_lp / battery.capacity_mwh * 100, 1),
            "soc_final_naive_pct": round(soc_naive / battery.capacity_mwh * 100, 1),
        })

    resultado = pd.DataFrame(filas)
    resultado["mejora_pct"] = (
        (resultado["ingreso_lp"] - resultado["ingreso_naive"])
        / resultado["ingreso_naive"].abs().clip(lower=1) * 100
    ).round(1)
    return resultado


if __name__ == "__main__":
    from hourly_shape import cargar_precio_horario_crudo

    df_h = cargar_precio_horario_crudo("../PrecioBolsa2026.xlsx")

    battery = BatteryConfig(
        capacity_mwh=4.0, power_mw=1.0, round_trip_efficiency=0.90,
        soc_min_pct=0.10, soc_max_pct=0.95, soc_init_pct=0.50,
        degradation_cost_per_mwh=3.0,
    )

    resultado = backtest_multi_dia(df_h, battery, dias=30)
    pd.set_option("display.width", 120)
    print(resultado.to_string(index=False))
    print()
    print(f"Ingreso total LP    (30 días): ${resultado['ingreso_lp'].sum():,.0f}")
    print(f"Ingreso total NAIVE (30 días): ${resultado['ingreso_naive'].sum():,.0f}")
    mejora_total = (resultado['ingreso_lp'].sum() / resultado['ingreso_naive'].sum() - 1) * 100
    print(f"Mejora del LP sobre naive: {mejora_total:.1f}%")
