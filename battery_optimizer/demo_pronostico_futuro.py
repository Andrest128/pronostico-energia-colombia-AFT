"""
demo_pronostico_futuro.py
────────────────────────────
Este es el script que amarra TODO el pipeline, tal como correría en
producción: a diferencia de backtest.py (que usa precios reales ya
ocurridos, "perfect foresight"), aquí usamos el PRONÓSTICO de Prophet
—el mismo modelo del dashboard— para generar un plan de despacho de los
próximos N días que AÚN NO HAN OCURRIDO.

Flujo:
  1. price_forecast.entrenar_y_pronosticar()  → pronóstico DIARIO (Prophet)
  2. hourly_shape.calcular_forma_horaria()     → forma horaria histórica
  3. hourly_forecast.construir_pronostico_horario() → pronóstico HORARIO
  4. optimizer.optimizar_despacho()            → plan de carga/descarga óptimo
     corrido en modo ROLLING: se optimiza día por día, encadenando el SOC,
     en vez de optimizar los N días de una sola vez. Esto es importante:
     en producción, cada día se vuelve a correr con el pronóstico
     actualizado (rolling horizon) — aquí se simula esa misma lógica.

Ejecutar:
  cd src && python3 demo_pronostico_futuro.py
"""

import sys
sys.path.insert(0, ".")
import pandas as pd

from battery_model import BatteryConfig
from price_forecast import entrenar_y_pronosticar
from hourly_shape import cargar_precio_horario_crudo, calcular_forma_horaria, resumen_spread_intradia
from hourly_forecast import construir_pronostico_horario
from optimizer import optimizar_despacho, resumen_resultado

DATA_DIR = "../data"
HORIZONTE_DIAS = 7  # días hacia adelante a planear


def main():
    print("=" * 70)
    print("1) Entrenando modelo Prophet de pronóstico DIARIO (igual al dashboard)")
    print("=" * 70)
    df, dp, forecast, modelo, fuentes, regresores, vars_extra = entrenar_y_pronosticar(
        DATA_DIR, horizonte_dias=HORIZONTE_DIAS
    )
    ultima_fecha_historica = dp["ds"].max()
    print(f"Último dato histórico: {ultima_fecha_historica.date()}")
    print(f"Fuentes de datos: {fuentes}")
    print(f"Regresores activos en el modelo: {regresores}")

    print()
    print("=" * 70)
    print("2) Calculando forma horaria a partir del histórico real")
    print("=" * 70)
    df_h = cargar_precio_horario_crudo(f"{DATA_DIR}/PrecioBolsa2026.xlsx")
    forma = calcular_forma_horaria(df_h, dias_recientes=180, por_tipo_dia=True)
    spread_info = resumen_spread_intradia(df_h, dias_recientes=90)
    print(f"Spread intradía reciente: {spread_info}")
    print("Forma horaria (semana), multiplicador vs. promedio del día:")
    print(forma["semana"].round(3).to_dict())

    print()
    print("=" * 70)
    print("3) Construyendo pronóstico HORARIO (Prophet diario × forma horaria)")
    print("=" * 70)
    fecha_inicio = ultima_fecha_historica + pd.Timedelta(days=1)
    pronostico_horario = construir_pronostico_horario(
        forecast, forma, fecha_inicio=fecha_inicio, horizonte_dias=HORIZONTE_DIAS
    )
    print(pronostico_horario.head(5).to_string(index=False))
    print(f"... ({len(pronostico_horario)} filas totales, {HORIZONTE_DIAS} días x 24h)")

    print()
    print("=" * 70)
    print("4) Optimizando despacho de la batería (rolling, día por día)")
    print("=" * 70)
    battery = BatteryConfig(
        capacity_mwh=4.0, power_mw=1.0, round_trip_efficiency=0.90,
        soc_min_pct=0.10, soc_max_pct=0.95, soc_init_pct=0.50,
        degradation_cost_per_mwh=3.0,
    )
    print(f"Batería simulada: {battery.capacity_mwh} MWh / {battery.power_mw} MW "
          f"(equivalente a {battery.capacity_mwh/battery.power_mw:.0f}h de autonomía)")

    soc_actual = battery.soc_init_mwh
    planes = []
    for dia_idx, fecha_dia in enumerate(pronostico_horario["fecha"].unique()):
        precios_dia = pronostico_horario.loc[
            pronostico_horario["fecha"] == fecha_dia, "precio_yhat"
        ].reset_index(drop=True)

        resultado = optimizar_despacho(precios_dia, battery, soc_inicial_mwh=soc_actual)
        resultado["fecha"] = fecha_dia
        planes.append(resultado)
        soc_actual = resultado["soc_mwh"].iloc[-1]

        resumen = resumen_resultado(resultado)
        print(f"  {fecha_dia}: ingreso proyectado ${resumen['ingreso_total']:,.0f} | "
              f"{resumen['horas_carga']}h carga @ ${resumen['precio_promedio_compra']:.0f} | "
              f"{resumen['horas_descarga']}h descarga @ ${resumen['precio_promedio_venta']:.0f}")

    plan_completo = pd.concat(planes, ignore_index=True)
    ingreso_total_horizonte = plan_completo["ingreso_hora"].sum()

    print()
    print("=" * 70)
    print(f"RESUMEN: ingreso proyectado total en {HORIZONTE_DIAS} días: ${ingreso_total_horizonte:,.0f}")
    print("=" * 70)
    print()
    print("⚠ IMPORTANTE — leer antes de usar estos números:")
    print("  Este plan usa el PRONÓSTICO de precio (con su margen de error propio")
    print("  de Prophet) combinado con una forma horaria HISTÓRICA PROMEDIO, no un")
    print("  pronóstico hora-por-hora real. Es una aproximación razonable para")
    print("  planear, pero en producción el plan debe RECALCULARSE cada día con")
    print("  el pronóstico actualizado (rolling horizon) — no ejecutar los 7 días")
    print("  de un tirón sin revisar. Ver README.md para más detalle sobre esta")
    print("  limitación y cómo migrarla a un pronóstico horario nativo.")

    plan_completo.to_csv("plan_despacho_futuro.csv", index=False)
    print()
    print("Plan detallado guardado en: plan_despacho_futuro.csv")


if __name__ == "__main__":
    main()
