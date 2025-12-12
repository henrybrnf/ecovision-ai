# Módulo Fuzzy Logic
"""
Sistema de inferencia difusa para evaluación de alertas.

Este módulo proporciona:
- AlertSystem: Sistema de alertas basado en lógica difusa
- FuzzyResult: Resultado de la evaluación difusa

El sistema evalúa situaciones basándose en:
- Cantidad de personas
- Velocidad de movimiento
- Densidad de zona

Y produce un nivel de alerta categorizado como:
- Normal (verde)
- Precaución (amarillo)
- Alerta (naranja)
- Emergencia (rojo)

Example:
    >>> from src.fuzzy_logic import AlertSystem
    >>> 
    >>> system = AlertSystem()
    >>> result = system.evaluate(person_count=10, movement_speed=0.8)
    >>> print(f"Alerta: {result.alert_category}")
"""

from .fuzzy_system import AlertSystem, FuzzyResult

__all__ = [
    "AlertSystem",
    "FuzzyResult"
]
