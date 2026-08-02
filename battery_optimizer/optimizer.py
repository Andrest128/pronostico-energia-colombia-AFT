"""
optimizer.py
──────────────
Motor de optimización del despacho de la batería (arbitraje de precio
de bolsa). Resuelve, para un horizonte de N horas, cuánto cargar y
cuánto descargar en cada hora para maximizar el ingreso neto, sujeto a
las restricciones físicas de la batería (BatteryConfig).

MODELO MATEMÁTICO (Programación Lineal)
-----------------------------------------
Variables de decisión, para cada hora t = 0..T-1:
  charge[t]    >= 0   MWh comprados de la red en la hora t
  discharge[t] >= 0   MWh vendidos a la red en la hora t
  soc[t]              MWh almacenados en la batería al FINAL de la hora t

Función objetivo (maximizar):
  ingreso = Σ ( precio[t] * discharge[t] - precio[t] * charge[t] )
            - Σ costo_degradacion_por_mwh * (charge[t] + discharge[t])

Restricciones:
  1. Balance de energía:
     soc[t] = soc[t-1] + charge[t] * eficiencia_carga - discharge[t]
     (con soc[-1] = soc_inicial)
  2. Límites de estado de carga:
     soc_min <= soc[t] <= soc_max   para todo t
  3. Límites de potencia (no se puede cargar/descargar más rápido que
     la potencia nominal de la batería en una hora):
     0 <= charge[t]    <= power_mw
     0 <= discharge[t] <= power_mw
  4. (Opcional/recomendada) No cargar y descargar en la misma hora
     simultáneamente — se logra de forma natural en la mayoría de
     soluciones óptimas porque hacerlo pierde dinero (pagas la
     ineficiencia dos veces), pero se puede forzar con variables
     binarias si se quiere garantía estricta (ver `forzar_no_simultaneo`).

Se usa PuLP con el solver CBC (incluido, gratuito, suficiente para este
tamaño de problema — decenas a cientos de variables).
"""

import pandas as pd
import pulp

from battery_model import BatteryConfig


def optimizar_despacho(
    precios: pd.Series,
    battery: BatteryConfig,
    soc_inicial_mwh: float = None,
    forzar_no_simultaneo: bool = False,
) -> pd.DataFrame:
    """
    Parámetros
    ----------
    precios : Series de precio ($/kWh o $/MWh — ver nota de unidades abajo),
        indexada 0..T-1 en orden cronológico horario.
    battery : configuración física de la batería.
    soc_inicial_mwh : estado de carga inicial. Si es None, usa battery.soc_init_mwh.
    forzar_no_simultaneo : si True, agrega variables binarias para prohibir
        cargar y descargar en la misma hora. Aumenta el tiempo de resolución
        (pasa de LP puro a MILP) — solo actívalo si ves en los resultados que
        el solver sí está cargando y descargando en la misma hora a la vez
        (raro, pero posible en horas con precio casi plano).

    NOTA DE UNIDADES: la función es agnóstica a si `precios` viene en
    $/kWh o $/MWh — el ingreso resultante estará en las mismas unidades
    monetarias por MWh que uses. El dashboard de pronóstico reporta en
    $/kWh; si quieres ingresos en pesos totales, multiplica precio en
    $/kWh por 1000 para pasarlo a $/MWh antes de llamar esta función
    (o interpreta el resultado como "miles de pesos").

    Retorna
    -------
    DataFrame con columnas: hora, precio, charge_mwh, discharge_mwh,
    soc_mwh, ingreso_hora, tipo_accion
    """
    T = len(precios)
    if T == 0:
        raise ValueError("La serie de precios está vacía.")

    soc_ini = battery.soc_init_mwh if soc_inicial_mwh is None else soc_inicial_mwh
    if not (battery.soc_min_mwh <= soc_ini <= battery.soc_max_mwh):
        raise ValueError(
            f"soc_inicial_mwh={soc_ini} fuera de rango "
            f"[{battery.soc_min_mwh}, {battery.soc_max_mwh}]"
        )

    prob = pulp.LpProblem("despacho_bateria", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=battery.power_mw)
    discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=battery.power_mw)
    soc = pulp.LpVariable.dicts(
        "soc", range(T), lowBound=battery.soc_min_mwh, upBound=battery.soc_max_mwh
    )

    precios_list = list(precios)

    # ── Función objetivo ──
    ingreso = pulp.lpSum(
        precios_list[t] * discharge[t] - precios_list[t] * charge[t]
        for t in range(T)
    )
    degradacion = pulp.lpSum(
        battery.degradation_cost_per_mwh * (charge[t] + discharge[t])
        for t in range(T)
    )
    prob += ingreso - degradacion

    # ── Restricción de balance de energía ──
    for t in range(T):
        soc_prev = soc_ini if t == 0 else soc[t - 1]
        prob += (
            soc[t] == soc_prev + charge[t] * battery.charge_efficiency - discharge[t]
        ), f"balance_hora_{t}"

    # ── (Opcional) prohibir carga y descarga simultánea ──
    if forzar_no_simultaneo:
        is_charging = pulp.LpVariable.dicts("is_charging", range(T), cat="Binary")
        for t in range(T):
            prob += charge[t] <= battery.power_mw * is_charging[t]
            prob += discharge[t] <= battery.power_mw * (1 - is_charging[t])

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(
            f"El optimizador no encontró una solución óptima. Estado: {pulp.LpStatus[status]}"
        )

    filas = []
    for t in range(T):
        c = charge[t].value() or 0.0
        d = discharge[t].value() or 0.0
        s = soc[t].value() or 0.0
        ingreso_hora = precios_list[t] * d - precios_list[t] * c - battery.degradation_cost_per_mwh * (c + d)
        tipo = "carga" if c > 0.01 else ("descarga" if d > 0.01 else "reposo")
        filas.append({
            "hora": t,
            "precio": precios_list[t],
            "charge_mwh": round(c, 4),
            "discharge_mwh": round(d, 4),
            "soc_mwh": round(s, 4),
            "soc_pct": round(s / battery.capacity_mwh * 100, 1),
            "ingreso_hora": round(ingreso_hora, 2),
            "tipo_accion": tipo,
        })

    return pd.DataFrame(filas)


def resumen_resultado(df_resultado: pd.DataFrame) -> dict:
    """Métricas agregadas del plan de despacho optimizado."""
    return {
        "ingreso_total": round(df_resultado["ingreso_hora"].sum(), 2),
        "energia_cargada_mwh": round(df_resultado["charge_mwh"].sum(), 2),
        "energia_descargada_mwh": round(df_resultado["discharge_mwh"].sum(), 2),
        "horas_carga": int((df_resultado["tipo_accion"] == "carga").sum()),
        "horas_descarga": int((df_resultado["tipo_accion"] == "descarga").sum()),
        "horas_reposo": int((df_resultado["tipo_accion"] == "reposo").sum()),
        "precio_promedio_compra": round(
            (df_resultado.loc[df_resultado["charge_mwh"] > 0.01, "precio"] *
             df_resultado.loc[df_resultado["charge_mwh"] > 0.01, "charge_mwh"]).sum() /
            max(df_resultado["charge_mwh"].sum(), 1e-9), 2
        ) if df_resultado["charge_mwh"].sum() > 0 else 0.0,
        "precio_promedio_venta": round(
            (df_resultado.loc[df_resultado["discharge_mwh"] > 0.01, "precio"] *
             df_resultado.loc[df_resultado["discharge_mwh"] > 0.01, "discharge_mwh"]).sum() /
            max(df_resultado["discharge_mwh"].sum(), 1e-9), 2
        ) if df_resultado["discharge_mwh"].sum() > 0 else 0.0,
    }
