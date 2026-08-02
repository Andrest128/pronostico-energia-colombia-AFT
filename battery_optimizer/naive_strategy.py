"""
naive_strategy.py
────────────────────
Estrategia de referencia SIN optimización: cargar en las horas más
baratas del día y descargar en las horas más caras, hasta donde la
potencia y capacidad de la batería lo permitan — sin usar programación
lineal, solo una regla simple basada en ranking de precios.

Sirve como punto de comparación para demostrar cuánto valor agrega el
optimizador LP frente a una heurística razonable pero simple (que es lo
que la mayoría de operadores hacen "a ojo" hoy sin herramienta).
"""

import math
import pandas as pd
from battery_model import BatteryConfig


def despacho_naive(precios: pd.Series, battery: BatteryConfig, soc_inicial_mwh: float = None) -> pd.DataFrame:
    """
    Regla: ordena las horas del día de más barata a más cara.
    Carga en las N horas más baratas necesarias para llenar la batería
    (a potencia máxima), descarga en las N horas más caras necesarias
    para vaciarla (a potencia máxima). Un ciclo completo por día.

    Es intencionalmente simple — no reoptimiza dentro del día ni
    considera el valor de "esperar" a un precio mejor. Por eso el
    optimizador LP normalmente le gana.

    NOTA (corregido): las horas de descarga necesarias se calculan sobre
    el SOC que la batería alcanza DESPUÉS de cargar (cerca de soc_max),
    no sobre el soc_inicial. Usar soc_inicial subestimaba las horas de
    descarga necesarias y hacía que esta estrategia de referencia se
    viera artificialmente peor de lo que en realidad es — un heurístico
    razonable sí vacía la batería casi por completo en el día, no se
    queda a medias por un error de cálculo.
    """
    T = len(precios)
    soc_ini = battery.soc_init_mwh if soc_inicial_mwh is None else soc_inicial_mwh

    horas_carga_necesarias = math.ceil(
        (battery.soc_max_mwh - soc_ini) / (battery.power_mw * battery.charge_efficiency)
    )
    soc_tras_carga = min(
        battery.soc_max_mwh,
        soc_ini + horas_carga_necesarias * battery.power_mw * battery.charge_efficiency
    )
    horas_descarga_necesarias = math.ceil(
        (soc_tras_carga - battery.soc_min_mwh) / battery.power_mw
    )

    orden_barato_a_caro = precios.sort_values().index.tolist()
    horas_carga = set(orden_barato_a_caro[:horas_carga_necesarias])
    orden_caro_a_barato = precios.sort_values(ascending=False).index.tolist()
    horas_descarga = set(orden_caro_a_barato[:horas_descarga_necesarias]) - horas_carga

    filas = []
    soc = soc_ini
    for t in range(T):
        c = d = 0.0
        if t in horas_carga and soc < battery.soc_max_mwh:
            c = min(battery.power_mw, (battery.soc_max_mwh - soc) / battery.charge_efficiency)
        elif t in horas_descarga and soc > battery.soc_min_mwh:
            d = min(battery.power_mw, soc - battery.soc_min_mwh)

        soc = soc + c * battery.charge_efficiency - d
        ingreso_hora = precios.iloc[t] * d - precios.iloc[t] * c - battery.degradation_cost_per_mwh * (c + d)
        tipo = "carga" if c > 0.01 else ("descarga" if d > 0.01 else "reposo")

        filas.append({
            "hora": t, "precio": precios.iloc[t],
            "charge_mwh": round(c, 4), "discharge_mwh": round(d, 4),
            "soc_mwh": round(soc, 4), "soc_pct": round(soc / battery.capacity_mwh * 100, 1),
            "ingreso_hora": round(ingreso_hora, 2), "tipo_accion": tipo,
        })

    return pd.DataFrame(filas)
