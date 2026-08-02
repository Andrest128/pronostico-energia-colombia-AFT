# Optimizador de despacho SAEB (baterías) — Colombia

Prototipo de motor de optimización que decide cuándo cargar y cuándo
descargar una batería (SAEB/BESS) para maximizar el ingreso por
arbitraje de precio de bolsa. Se integra con el modelo de pronóstico
de precio de bolsa ya existente en
[pronostico-energia-colombia-AFT](https://github.com/Andrest128/pronostico-energia-colombia-AFT)
(Prophet + Streamlit), sin modificarlo.

## Por qué existe este proyecto

El dashboard de pronóstico original predice el precio **promedio diario**
de bolsa. Eso es correcto para ver tendencia y alertas de precio, pero
el valor de una batería viene de la diferencia de precio **dentro** del
día (comprar barato de madrugada, vender caro en la punta de la noche).
Con los datos reales del repo, el spread intradía (máx-mín del día)
representa en promedio ~120-130% del precio promedio diario en los
últimos 90 días — es una señal demasiado grande para dejarla sobre la
mesa.

Este proyecto no reemplaza el dashboard: lo **extiende**, reutilizando
exactamente el mismo modelo Prophet, y le agrega una capa de:
1. Desagregación horaria del pronóstico diario (forma horaria histórica)
2. Un modelo físico simple de la batería
3. Un optimizador de programación lineal (LP) que decide el despacho óptimo

## Estructura del proyecto

```
battery_project/
├── data/                          # Excels originales del repo (no versionados aquí)
│   ├── PrecioBolsa2026.xlsx
│   ├── Aportes2026.xlsx
│   ├── Embalses2026.xlsx
│   ├── Demanda2026.xlsx
│   └── ONI.xlsx
├── src/
│   ├── battery_model.py           # Parámetros físicos de la batería (BatteryConfig)
│   ├── price_forecast.py          # Carga de datos + Prophet, sin dependencia de Streamlit
│   ├── hourly_shape.py            # Calcula la forma horaria histórica del precio
│   ├── hourly_forecast.py         # Combina pronóstico diario × forma horaria = pronóstico horario
│   ├── optimizer.py               # Motor de optimización LP (PuLP) del despacho
│   ├── naive_strategy.py          # Estrategia de referencia sin optimización (para comparar)
│   ├── backtest.py                # Backtest multi-día con precios reales (perfect foresight)
│   └── demo_pronostico_futuro.py  # Pipeline completo: pronóstico → plan de despacho futuro
├── requirements.txt
└── README.md
```

## Cómo correrlo

```bash
pip install -r requirements.txt

# Opción A: backtest contra precios históricos reales (mide el techo teórico)
cd src && python3 backtest.py

# Opción B: pipeline completo con pronóstico Prophet real hacia el futuro
cd src && python3 demo_pronostico_futuro.py
```

`demo_pronostico_futuro.py` es el que representa el flujo de producción:
entrena Prophet, calcula la forma horaria, arma el pronóstico horario y
optimiza el despacho día por día (rolling horizon), guardando el
resultado en `plan_despacho_futuro.csv`.

## El modelo matemático (resumen)

Para cada hora *t* del horizonte, el optimizador decide `charge[t]` y
`discharge[t]` (MWh) para maximizar:

```
ingreso = Σ (precio[t]·discharge[t] − precio[t]·charge[t]) − Σ costo_degradación·(charge[t]+discharge[t])
```

sujeto a:
- Balance de energía: `soc[t] = soc[t-1] + charge[t]·eficiencia − discharge[t]`
- Límites de SOC: `soc_min ≤ soc[t] ≤ soc_max`
- Límites de potencia: `0 ≤ charge[t], discharge[t] ≤ power_mw`

Se resuelve con PuLP + CBC (solver gratuito, suficiente para este
tamaño de problema — 24-168 variables típicamente).

Ver los docstrings de `optimizer.py` y `battery_model.py` para el
detalle completo de cada supuesto y por qué se modeló así.

## Resultados de validación (con datos reales del repo)

**Backtest de 30 días** (23 abril – 28 mayo 2026, precios reales,
batería de 4 MWh / 1 MW, eficiencia 90%):
- Ingreso LP: $31,277 vs. Ingreso naive: $29,166 → **+7.2%**

Este número es más modesto de lo que se podría esperar, y es
intencional dejarlo así de honesto: con solo un ciclo de carga/descarga
posible por día y sin restricciones de potencia muy ajustadas, la
regla naive (cargar en las horas más baratas, vender en las más caras)
ya captura la mayor parte del valor — coincide exactamente con el
óptimo en varios días del backtest. La ventana donde el LP realmente
se distancia es: baterías con relación potencia/capacidad que impide
un ciclo completo en pocas horas, escenarios con más de un valle/pico
de precio por día, y el manejo explícito del costo de degradación
(el naive no lo considera al decidir cuánto ciclar).

## Limitaciones conocidas (documentadas a propósito)

1. **Forma horaria como patrón promedio, no pronóstico real.** El
   pronóstico horario se construye multiplicando el pronóstico diario
   de Prophet por un patrón horario histórico promedio (últimos 180
   días). Esto asume que la forma del día se mantiene razonablemente
   estable. Si el despacho cambia estructuralmente (ej. entra mucha
   solar nueva y desplaza el pico), hay que recalcular la forma con
   datos más recientes.
2. **`price_forecast.py` duplica lógica de `app.py`.** Es intencional
   y temporal: se separó la lógica de Streamlit para poder reutilizarla
   fuera del dashboard, sin modificar el archivo original en producción.
   Refactor sugerido a futuro: mover la lógica de `cargar_datos()` y
   `entrenar_y_pronosticar()` de `app.py` a este módulo, y que `app.py`
   quede como una capa delgada de UI que importa de aquí — así se
   elimina la duplicación y ambos quedan siempre sincronizados.
3. **Costo de degradación es un proxy simple ($/MWh movido), no un
   modelo de ciclos de vida real.** Ajustar `degradation_cost_per_mwh`
   en `BatteryConfig` con datos reales de la ficha técnica del
   fabricante (costo de reemplazo ÷ ciclos de vida garantizados) en
   cuanto se tenga un caso concreto.
4. **No modela servicios de reserva/AGC**, solo arbitraje de precio de
   bolsa. Es la fase 2 sugerida (ver conversación previa).
5. **Backtest usa "perfect foresight"** (precios reales ya ocurridos),
   por diseño — sirve para medir el techo teórico de ingresos, no el
   desempeño esperado con pronóstico. El desempeño con pronóstico real
   será menor en la medida del error de Prophet; `demo_pronostico_futuro.py`
   es la versión honesta de "lo que sabríamos de antemano".

## Próximos pasos sugeridos

- Correr `demo_pronostico_futuro.py` regularmente (ej. diario, vía cron
  o GitHub Actions) y comparar el ingreso proyectado vs. el realmente
  obtenido, para medir el error real del pronóstico aplicado al negocio
  (no solo el MAPE genérico del dashboard).
- Sensibilizar el tamaño de batería (`capacity_mwh`, `power_mw`) contra
  el backtest para dimensionar el caso de negocio con distintos activos.
- Si se consigue el dato real de un SAEB instalado en Colombia (ej.
  Termozipa o Celsia Palmira), correr el backtest con sus parámetros
  reales como caso de estudio de venta.
