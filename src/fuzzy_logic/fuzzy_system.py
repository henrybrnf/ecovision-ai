"""
Sistema de inferencia difusa para evaluación de alertas.

Este módulo implementa un sistema de lógica difusa que evalúa
el nivel de alerta basándose en múltiples variables de entrada
como cantidad de personas, velocidad de movimiento, etc.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class FuzzyResult:
    """Resultado del sistema difuso."""
    alert_level: float          # Nivel de alerta (0.0 - 1.0)
    alert_category: str         # Categoría: "normal", "precaución", "alerta", "emergencia"
    inputs: Dict[str, float]    # Valores de entrada usados
    
    @property
    def color(self) -> Tuple[int, int, int]:
        """Color BGR para visualización."""
        colors = {
            "normal": (0, 255, 0),      # Verde
            "precaución": (0, 255, 255), # Amarillo
            "alerta": (0, 165, 255),    # Naranja
            "emergencia": (0, 0, 255)   # Rojo
        }
        return colors.get(self.alert_category, (128, 128, 128))


class AlertSystem:
    """
    Sistema de alertas basado en lógica difusa.
    
    Este sistema evalúa el nivel de peligro de una situación
    basándose en variables como:
    - Cantidad de personas detectadas
    - Velocidad promedio de movimiento
    - Densidad en zonas específicas
    
    La salida es un nivel de alerta continuo (0.0 - 1.0) que
    se categoriza como: normal, precaución, alerta, emergencia.
    
    Attributes:
        person_count: Variable de entrada - cantidad de personas
        movement_speed: Variable de entrada - velocidad de movimiento
        zone_density: Variable de entrada - densidad de zona
        alert_level: Variable de salida - nivel de alerta
        
    Example:
        >>> alert_system = AlertSystem()
        >>> result = alert_system.evaluate(person_count=5, movement_speed=0.7)
        >>> print(f"Alerta: {result.alert_category} ({result.alert_level:.2f})")
    """
    
    def __init__(
        self,
        max_persons: int = 20,
        defuzzify_method: str = "centroid"
    ):
        """
        Inicializa el sistema de alertas difuso.
        
        Args:
            max_persons: Número máximo de personas esperado
            defuzzify_method: Método de defuzzificación
        """
        print("🌀 Inicializando sistema de lógica difusa...")
        
        self.max_persons = max_persons
        self.defuzzify_method = defuzzify_method
        
        # Crear variables de entrada y salida
        self._create_variables()
        
        # Crear funciones de membresía
        self._create_membership_functions()
        
        # Crear reglas difusas
        self._create_rules()
        
        # Crear sistema de control
        self._create_control_system()
        
        print("✅ Sistema difuso inicializado")
    
    def _create_variables(self):
        """Crea las variables lingüísticas del sistema."""
        # Variable de entrada: cantidad de personas (0 a max_persons)
        self.person_count = ctrl.Antecedent(
            np.arange(0, self.max_persons + 1, 1),
            'person_count'
        )
        
        # Variable de entrada: velocidad de movimiento (0 a 1)
        self.movement_speed = ctrl.Antecedent(
            np.arange(0, 1.01, 0.01),
            'movement_speed'
        )
        
        # Variable de entrada: densidad de zona (0 a 1)
        self.zone_density = ctrl.Antecedent(
            np.arange(0, 1.01, 0.01),
            'zone_density'
        )
        
        # Variable de salida: nivel de alerta (0 a 1)
        self.alert_level = ctrl.Consequent(
            np.arange(0, 1.01, 0.01),
            'alert_level',
            defuzzify_method=self.defuzzify_method
        )
    
    def _create_membership_functions(self):
        """Crea las funciones de membresía para cada variable."""
        
        # Funciones de membresía para CANTIDAD DE PERSONAS
        # Pocas: 0-5, Moderadas: 3-12, Muchas: 10+
        self.person_count['pocas'] = fuzz.trapmf(
            self.person_count.universe, [0, 0, 3, 6]
        )
        self.person_count['moderadas'] = fuzz.trimf(
            self.person_count.universe, [4, 8, 12]
        )
        self.person_count['muchas'] = fuzz.trapmf(
            self.person_count.universe, [10, 14, self.max_persons, self.max_persons]
        )
        
        # Funciones de membresía para VELOCIDAD DE MOVIMIENTO
        # Lento: 0-0.3, Moderado: 0.2-0.7, Rápido: 0.5-1.0
        self.movement_speed['lento'] = fuzz.trapmf(
            self.movement_speed.universe, [0, 0, 0.2, 0.4]
        )
        self.movement_speed['moderado'] = fuzz.trimf(
            self.movement_speed.universe, [0.25, 0.5, 0.75]
        )
        self.movement_speed['rapido'] = fuzz.trapmf(
            self.movement_speed.universe, [0.6, 0.8, 1.0, 1.0]
        )
        
        # Funciones de membresía para DENSIDAD DE ZONA
        # Baja: 0-0.3, Media: 0.2-0.7, Alta: 0.5-1.0
        self.zone_density['baja'] = fuzz.trapmf(
            self.zone_density.universe, [0, 0, 0.2, 0.4]
        )
        self.zone_density['media'] = fuzz.trimf(
            self.zone_density.universe, [0.25, 0.5, 0.75]
        )
        self.zone_density['alta'] = fuzz.trapmf(
            self.zone_density.universe, [0.6, 0.8, 1.0, 1.0]
        )
        
        # Funciones de membresía para NIVEL DE ALERTA
        # Normal: 0-0.25, Precaución: 0.2-0.5, Alerta: 0.4-0.75, Emergencia: 0.7-1.0
        self.alert_level['normal'] = fuzz.trapmf(
            self.alert_level.universe, [0, 0, 0.15, 0.3]
        )
        self.alert_level['precaucion'] = fuzz.trimf(
            self.alert_level.universe, [0.2, 0.4, 0.55]
        )
        self.alert_level['alerta'] = fuzz.trimf(
            self.alert_level.universe, [0.45, 0.65, 0.8]
        )
        self.alert_level['emergencia'] = fuzz.trapmf(
            self.alert_level.universe, [0.7, 0.85, 1.0, 1.0]
        )
    
    def _create_rules(self):
        """Crea las reglas difusas IF-THEN."""
        
        # Reglas basadas en cantidad de personas
        self.rules = []
        
        # Regla 1: Si hay pocas personas Y movimiento lento → Normal
        self.rules.append(ctrl.Rule(
            self.person_count['pocas'] & self.movement_speed['lento'],
            self.alert_level['normal']
        ))
        
        # Regla 2: Si hay pocas personas Y movimiento moderado → Normal
        self.rules.append(ctrl.Rule(
            self.person_count['pocas'] & self.movement_speed['moderado'],
            self.alert_level['normal']
        ))
        
        # Regla 3: Si hay pocas personas Y movimiento rápido → Precaución
        self.rules.append(ctrl.Rule(
            self.person_count['pocas'] & self.movement_speed['rapido'],
            self.alert_level['precaucion']
        ))
        
        # Regla 4: Si hay moderadas personas Y movimiento lento → Normal
        self.rules.append(ctrl.Rule(
            self.person_count['moderadas'] & self.movement_speed['lento'],
            self.alert_level['normal']
        ))
        
        # Regla 5: Si hay moderadas personas Y movimiento moderado → Precaución
        self.rules.append(ctrl.Rule(
            self.person_count['moderadas'] & self.movement_speed['moderado'],
            self.alert_level['precaucion']
        ))
        
        # Regla 6: Si hay moderadas personas Y movimiento rápido → Alerta
        self.rules.append(ctrl.Rule(
            self.person_count['moderadas'] & self.movement_speed['rapido'],
            self.alert_level['alerta']
        ))
        
        # Regla 7: Si hay muchas personas Y movimiento lento → Precaución
        self.rules.append(ctrl.Rule(
            self.person_count['muchas'] & self.movement_speed['lento'],
            self.alert_level['precaucion']
        ))
        
        # Regla 8: Si hay muchas personas Y movimiento moderado → Alerta
        self.rules.append(ctrl.Rule(
            self.person_count['muchas'] & self.movement_speed['moderado'],
            self.alert_level['alerta']
        ))
        
        # Regla 9: Si hay muchas personas Y movimiento rápido → Emergencia
        self.rules.append(ctrl.Rule(
            self.person_count['muchas'] & self.movement_speed['rapido'],
            self.alert_level['emergencia']
        ))
        
        # Reglas adicionales con densidad de zona
        # Regla 10: Si densidad alta Y movimiento rápido → Emergencia
        self.rules.append(ctrl.Rule(
            self.zone_density['alta'] & self.movement_speed['rapido'],
            self.alert_level['emergencia']
        ))
        
        # Regla 11: Si densidad alta Y muchas personas → Emergencia
        self.rules.append(ctrl.Rule(
            self.zone_density['alta'] & self.person_count['muchas'],
            self.alert_level['emergencia']
        ))
    
    def _create_control_system(self):
        """Crea el sistema de control difuso."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def evaluate(
        self,
        person_count: int = 0,
        movement_speed: float = 0.0,
        zone_density: float = 0.0
    ) -> FuzzyResult:
        """
        Evalúa el nivel de alerta basado en las entradas.
        
        Args:
            person_count: Cantidad de personas detectadas
            movement_speed: Velocidad de movimiento normalizada (0.0 - 1.0)
            zone_density: Densidad de la zona (0.0 - 1.0)
        
        Returns:
            FuzzyResult con el nivel de alerta y categoría
        """
        # Limitar valores a rangos válidos
        person_count = max(0, min(person_count, self.max_persons))
        movement_speed = max(0.0, min(movement_speed, 1.0))
        zone_density = max(0.0, min(zone_density, 1.0))
        
        # Asignar valores a las entradas
        self.simulation.input['person_count'] = person_count
        self.simulation.input['movement_speed'] = movement_speed
        self.simulation.input['zone_density'] = zone_density
        
        # Ejecutar evaluación
        try:
            self.simulation.compute()
            alert_value = self.simulation.output['alert_level']
        except Exception:
            # Si falla (por ejemplo, reglas no cubren el caso), usar valor por defecto
            alert_value = 0.1
        
        # Categorizar el nivel de alerta
        category = self._categorize_alert(alert_value)
        
        return FuzzyResult(
            alert_level=alert_value,
            alert_category=category,
            inputs={
                "person_count": person_count,
                "movement_speed": movement_speed,
                "zone_density": zone_density
            }
        )
    
    def _categorize_alert(self, alert_value: float) -> str:
        """Categoriza el valor de alerta en una etiqueta."""
        if alert_value < 0.25:
            return "normal"
        elif alert_value < 0.5:
            return "precaución"
        elif alert_value < 0.75:
            return "alerta"
        else:
            return "emergencia"
    
    def get_membership_degrees(
        self,
        person_count: int,
        movement_speed: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Obtiene los grados de membresía para las entradas dadas.
        
        Returns:
            Diccionario con los grados de membresía de cada variable
        """
        pc_universe = self.person_count.universe
        ms_universe = self.movement_speed.universe
        
        # Encontrar índices más cercanos
        pc_idx = np.argmin(np.abs(pc_universe - person_count))
        ms_idx = np.argmin(np.abs(ms_universe - movement_speed))
        
        return {
            "person_count": {
                "pocas": float(fuzz.interp_membership(
                    pc_universe, self.person_count['pocas'].mf, person_count
                )),
                "moderadas": float(fuzz.interp_membership(
                    pc_universe, self.person_count['moderadas'].mf, person_count
                )),
                "muchas": float(fuzz.interp_membership(
                    pc_universe, self.person_count['muchas'].mf, person_count
                ))
            },
            "movement_speed": {
                "lento": float(fuzz.interp_membership(
                    ms_universe, self.movement_speed['lento'].mf, movement_speed
                )),
                "moderado": float(fuzz.interp_membership(
                    ms_universe, self.movement_speed['moderado'].mf, movement_speed
                )),
                "rapido": float(fuzz.interp_membership(
                    ms_universe, self.movement_speed['rapido'].mf, movement_speed
                ))
            }
        }
    
    def visualize_membership_functions(self, save_path: Optional[str] = None):
        """
        Visualiza las funciones de membresía.
        
        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cantidad de personas
        ax = axes[0, 0]
        ax.plot(self.person_count.universe, self.person_count['pocas'].mf, 
                'g', label='Pocas')
        ax.plot(self.person_count.universe, self.person_count['moderadas'].mf, 
                'y', label='Moderadas')
        ax.plot(self.person_count.universe, self.person_count['muchas'].mf, 
                'r', label='Muchas')
        ax.set_title('Cantidad de Personas')
        ax.set_xlabel('Personas')
        ax.set_ylabel('Membresía')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocidad de movimiento
        ax = axes[0, 1]
        ax.plot(self.movement_speed.universe, self.movement_speed['lento'].mf, 
                'g', label='Lento')
        ax.plot(self.movement_speed.universe, self.movement_speed['moderado'].mf, 
                'y', label='Moderado')
        ax.plot(self.movement_speed.universe, self.movement_speed['rapido'].mf, 
                'r', label='Rápido')
        ax.set_title('Velocidad de Movimiento')
        ax.set_xlabel('Velocidad (normalizada)')
        ax.set_ylabel('Membresía')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Densidad de zona
        ax = axes[1, 0]
        ax.plot(self.zone_density.universe, self.zone_density['baja'].mf, 
                'g', label='Baja')
        ax.plot(self.zone_density.universe, self.zone_density['media'].mf, 
                'y', label='Media')
        ax.plot(self.zone_density.universe, self.zone_density['alta'].mf, 
                'r', label='Alta')
        ax.set_title('Densidad de Zona')
        ax.set_xlabel('Densidad (normalizada)')
        ax.set_ylabel('Membresía')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Nivel de alerta
        ax = axes[1, 1]
        ax.plot(self.alert_level.universe, self.alert_level['normal'].mf, 
                'g', label='Normal')
        ax.plot(self.alert_level.universe, self.alert_level['precaucion'].mf, 
                'y', label='Precaución')
        ax.plot(self.alert_level.universe, self.alert_level['alerta'].mf, 
                'orange', label='Alerta')
        ax.plot(self.alert_level.universe, self.alert_level['emergencia'].mf, 
                'r', label='Emergencia')
        ax.set_title('Nivel de Alerta (Salida)')
        ax.set_xlabel('Nivel de Alerta')
        ax.set_ylabel('Membresía')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Gráfico guardado en: {save_path}")
        
        plt.close()


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test del Sistema de Lógica Difusa")
    print("=" * 50)
    
    # Crear sistema
    alert_system = AlertSystem()
    
    # Probar diferentes escenarios
    test_cases = [
        {"person_count": 2, "movement_speed": 0.2, "expected": "normal"},
        {"person_count": 5, "movement_speed": 0.5, "expected": "precaución"},
        {"person_count": 10, "movement_speed": 0.7, "expected": "alerta"},
        {"person_count": 15, "movement_speed": 0.9, "expected": "emergencia"},
    ]
    
    print("\n📊 Evaluando escenarios de prueba:")
    print("-" * 50)
    
    for case in test_cases:
        result = alert_system.evaluate(
            person_count=case["person_count"],
            movement_speed=case["movement_speed"]
        )
        
        status = "✅" if result.alert_category == case["expected"] else "⚠️"
        print(f"{status} Personas: {case['person_count']:2d}, "
              f"Velocidad: {case['movement_speed']:.1f} → "
              f"{result.alert_category.upper():12s} ({result.alert_level:.2f})")
    
    # Guardar visualización
    print("\n📈 Generando visualización de funciones de membresía...")
    alert_system.visualize_membership_functions("outputs/fuzzy_membership.png")
    
    print("\n✅ Test completado exitosamente!")
