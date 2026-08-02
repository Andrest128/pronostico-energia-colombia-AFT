"""
battery_model.py
─────────────────
Define los parámetros físicos y restricciones de un sistema de
almacenamiento con baterías (SAEB/BESS). Es la única fuente de verdad
sobre "qué puede hacer la batería" — el optimizador y el backtest
importan esta clase, nunca hardcodean un número de capacidad o potencia.

Todas las unidades de energía están en MWh y de potencia en MW,
salvo que se indique lo contrario.
"""

from dataclasses import dataclass


@dataclass
class BatteryConfig:
    capacity_mwh: float          # Capacidad nominal utilizable de la batería
    power_mw: float               # Potencia máxima de carga/descarga (MW)
    round_trip_efficiency: float = 0.90   # Eficiencia ida y vuelta (0-1). Ej: 0.90 = 90%
    soc_min_pct: float = 0.10     # Límite inferior de estado de carga (protege la batería)
    soc_max_pct: float = 0.95     # Límite superior de estado de carga
    soc_init_pct: float = 0.50    # Estado de carga inicial al arrancar la simulación
    degradation_cost_per_mwh: float = 3.0  # Costo "de desgaste" por MWh de energía movida
    # (cargada + descargada). Es un proxy simple del costo de ciclos de vida
    # de la batería — desincentiva al optimizador de mover energía "por moverla"
    # cuando el margen de precio no lo justifica. Ajustable según ficha técnica
    # real del fabricante (costo de reemplazo / ciclos de vida garantizados).

    def __post_init__(self):
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh debe ser mayor que 0")
        if self.power_mw <= 0:
            raise ValueError("power_mw debe ser mayor que 0")
        if not (0 < self.round_trip_efficiency <= 1):
            raise ValueError("round_trip_efficiency debe estar entre 0 y 1")
        if not (0 <= self.soc_min_pct < self.soc_max_pct <= 1):
            raise ValueError("Se requiere 0 <= soc_min_pct < soc_max_pct <= 1")
        if not (self.soc_min_pct <= self.soc_init_pct <= self.soc_max_pct):
            raise ValueError("soc_init_pct debe estar entre soc_min_pct y soc_max_pct")

    @property
    def soc_min_mwh(self) -> float:
        return self.capacity_mwh * self.soc_min_pct

    @property
    def soc_max_mwh(self) -> float:
        return self.capacity_mwh * self.soc_max_pct

    @property
    def soc_init_mwh(self) -> float:
        return self.capacity_mwh * self.soc_init_pct

    @property
    def charge_efficiency(self) -> float:
        """
        Se modela toda la pérdida de eficiencia ida-y-vuelta en el lado de
        carga (simplificación estándar en modelos de despacho de baterías):
        de 1 MWh comprado de la red, solo entra `charge_efficiency` MWh al
        estado de carga. La descarga hacia la red se asume 1:1.
        """
        return self.round_trip_efficiency

    @property
    def hours_to_full_from_empty(self) -> float:
        """Horas necesarias para cargar de soc_min a soc_max a potencia máxima."""
        return (self.soc_max_mwh - self.soc_min_mwh) / self.power_mw
