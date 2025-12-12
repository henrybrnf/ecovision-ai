"""
Simulación del ecosistema evolutivo.

Este módulo orquesta la simulación completa:
- Gestiona la población de agentes
- Ejecuta el ciclo de vida (percepción, decisión, acción)
- Aplica el algoritmo genético para evolución
- Integra con el sistema de detección y alertas
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

from .agent import Agent, AgentConfig
from .genetics import GeneticAlgorithm, GeneticConfig


@dataclass
class SimulationConfig:
    """Configuración de la simulación."""
    world_width: int = 800          # Ancho del mundo
    world_height: int = 600         # Alto del mundo
    agent_count: int = 20           # Número de agentes
    steps_per_generation: int = 500 # Pasos por generación
    max_generations: int = 100      # Máximo de generaciones
    food_count: int = 30            # Cantidad de comida
    food_respawn_rate: float = 0.5  # Probabilidad de respawn por frame
    

class Simulation:
    """
    Simulación del ecosistema de agentes evolutivos.
    
    Esta clase maneja:
    - La población de agentes
    - El ciclo de simulación
    - La evolución mediante algoritmo genético
    - La integración con detecciones externas
    
    Attributes:
        config: Configuración de la simulación
        agents: Lista de agentes activos
        genetic_algorithm: Algoritmo genético
        generation: Generación actual
        step: Paso actual dentro de la generación
    
    Example:
        >>> sim = Simulation()
        >>> sim.start()
        >>> while not sim.is_done():
        ...     sim.update(detected_objects, alert_level)
        ...     # Renderizar...
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        genetic_config: Optional[GeneticConfig] = None
    ):
        """
        Inicializa la simulación.
        
        Args:
            config: Configuración de la simulación
            genetic_config: Configuración del algoritmo genético
        """
        self.config = config or SimulationConfig()
        
        # Configurar algoritmo genético
        ga_config = genetic_config or GeneticConfig(
            population_size=self.config.agent_count
        )
        self.genetic_algorithm = GeneticAlgorithm(ga_config)
        
        # Estado
        self.agents: List[Agent] = []
        self.generation = 0
        self.step = 0
        self.running = False
        self.paused = False
        
        # Datos actuales
        self.current_detections: List[Tuple[float, float]] = []
        self.current_detections: List[Tuple[float, float]] = []
        self.current_alert_level: float = 0.0
        self.food_items: List[Tuple[float, float]] = [] # Lista de comida (x, y)
        
        # Estadísticas
        self.start_time = 0.0
        self.total_steps = 0
        
        print(f"🌍 Simulación inicializada:")
        print(f"   Mundo: {self.config.world_width}x{self.config.world_height}")
        print(f"   Agentes: {self.config.agent_count}")
        print(f"   Pasos/gen: {self.config.steps_per_generation}")
    
    @property
    def world_size(self) -> Tuple[int, int]:
        """Tamaño del mundo."""
        return (self.config.world_width, self.config.world_height)
    
    def start(self):
        """Inicia la simulación creando la población inicial."""
        print("\n🚀 Iniciando simulación...")
        
        # Crear población
        self.agents = self.genetic_algorithm.create_population(self.world_size)
        
        self.generation = 1
        self.step = 0
        self.running = True
        self.start_time = time.time()
        
        # Generar comida inicial
        self.food_items = []
        for _ in range(self.config.food_count):
            self._spawn_food()
            
        print(f"✅ Simulación iniciada - Generación {self.generation}")
    
    def stop(self):
        """Detiene la simulación."""
        self.running = False
        elapsed = time.time() - self.start_time
        print(f"\n⏹️ Simulación detenida")
        print(f"   Tiempo total: {elapsed:.1f}s")
        print(f"   Generaciones: {self.generation}")
        print(f"   Pasos totales: {self.total_steps}")
    
    def toggle_pause(self):
        """Pausa o reanuda la simulación."""
        self.paused = not self.paused
        status = "⏸️ Pausada" if self.paused else "▶️ Reanudada"
        print(f"{status}")
    
    def is_done(self) -> bool:
        """Verifica si la simulación debe terminar."""
        return (
            not self.running or
            self.generation > self.config.max_generations
        )
    
    def update(
        self,
        detected_objects: Optional[List[Tuple[float, float]]] = None,
        alert_level: float = 0.0
    ):
        """
        Actualiza la simulación un paso.
        
        Args:
            detected_objects: Posiciones de objetos detectados
            alert_level: Nivel de alerta actual (0.0 - 1.0)
        """
        if not self.running or self.paused:
            return
        
        # Guardar datos actuales
        self.current_detections = detected_objects or []
        self.current_alert_level = alert_level
        
        # Mapear detecciones al espacio del ecosistema si es necesario
        mapped_detections = self._map_detections_to_world(detected_objects)
        
        # 0. Gestionar Comida
        if len(self.food_items) < self.config.food_count:
            if np.random.random() < self.config.food_respawn_rate:
                self._spawn_food()

        # Actualizar cada agente
        active_agents = [a for a in self.agents if a.alive]
        
        for agent in active_agents:
            # 1. Percibir (Ahora incluye comida)
            agent.perceive(
                detected_objects=mapped_detections,
                food_items=self.food_items,
                alert_level=alert_level,
                other_agents=self.agents
            )
            
            # 2. Decidir
            action = agent.decide()
            
            # 3. Actuar
            agent.act(action)
            
            # 4. Actualizar posición
            agent.update()
            
            # 4b. Verificar colisión con comida
            if agent.energy < agent.config.max_energy:
                # Buscar comida cercana
                for i, food in enumerate(self.food_items):
                    dist = np.linalg.norm(agent.position - np.array(food))
                    if dist < (agent.config.size + 5): # Colisión
                        agent.eat()
                        self.food_items.pop(i)
                        break
            
            # 5. Calcular fitness
            agent.calculate_fitness(mapped_detections, alert_level)
        
        self.step += 1
        self.total_steps += 1
        
        # Verificar si es momento de evolucionar
        if self.step >= self.config.steps_per_generation:
            self._evolve()
    
    def _map_detections_to_world(
        self,
        detections: Optional[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """
        Mapea las detecciones del video al espacio del ecosistema.
        
        Las detecciones vienen en coordenadas de video (ej: 640x480)
        y se mapean al espacio del mundo (ej: 800x600).
        """
        if not detections:
            return []
        
        # Por ahora, asumimos que ya están en el espacio correcto
        # o aplicamos una transformación simple
        mapped = []
        for x, y in detections:
            # Escalar si es necesario (asumiendo video de 640x480)
            scale_x = self.config.world_width / 640
            scale_y = self.config.world_height / 480
            mapped.append((x * scale_x, y * scale_y))
        
        return mapped
    
    def _evolve(self):
        """Evoluciona la población a la siguiente generación."""
        # Obtener estadísticas antes de evolucionar
        best_agent = self.genetic_algorithm.get_best_agent(self.agents)
        avg_fitness = np.mean([a.fitness for a in self.agents])
        
        print(f"\n🧬 Generación {self.generation} completada:")
        print(f"   Mejor fitness: {best_agent.fitness:.2f}")
        print(f"   Fitness promedio: {avg_fitness:.2f}")
        
        # Evolucionar
        self.agents = self.genetic_algorithm.evolve(self.agents)
        
        self.generation += 1
        self.step = 0
        
        print(f"✅ Nueva generación {self.generation} creada")
    
    def get_best_agent(self) -> Agent:
        """Retorna el mejor agente actual."""
        return self.genetic_algorithm.get_best_agent(self.agents)
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas de la simulación.
        
        Returns:
            Diccionario con estadísticas
        """
        ga_stats = self.genetic_algorithm.get_statistics()
        
        return {
            "generation": self.generation,
            "step": self.step,
            "total_steps": self.total_steps,
            "steps_per_gen": self.config.steps_per_generation,
            "progress_in_gen": self.step / self.config.steps_per_generation,
            "running": self.running,
            "paused": self.paused,
            "agent_count": len(self.agents),
            "current_alert": self.current_alert_level,
            "detections_count": len(self.current_detections),
            **ga_stats
        }
    
    def _spawn_food(self):
        """Genera comida en posición aleatoria."""
        margin = 20
        pos = (
            np.random.uniform(margin, self.config.world_width - margin),
            np.random.uniform(margin, self.config.world_height - margin)
        )
        self.food_items.append(pos)
        
    def get_food_positions(self) -> List[Tuple[float, float]]:
        return self.food_items
    
    def reset(self):
        """Reinicia la simulación desde cero."""
        print("\n🔄 Reiniciando simulación...")
        
        # Resetear algoritmo genético
        self.genetic_algorithm.generation = 0
        self.genetic_algorithm.best_fitness_history.clear()
        self.genetic_algorithm.avg_fitness_history.clear()
        
        # Reiniciar estado
        self.generation = 0
        self.step = 0
        self.total_steps = 0
        self.running = False
        self.paused = False
        
        # Crear nueva población
        self.start()
    
    def save_state(self, filepath: str):
        """Guarda el estado actual de la simulación."""
        self.genetic_algorithm.save_best_genome(self.agents, filepath)
    
    def get_agent_positions(self) -> List[Tuple[float, float, int]]:
        """
        Obtiene las posiciones de todos los agentes.
        
        Returns:
            Lista de (x, y, id)
        """
        return [
            (float(a.position[0]), float(a.position[1]), a.id)
            for a in self.agents
        ]


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test de la Simulación")
    print("=" * 50)
    
    # Crear simulación
    config = SimulationConfig(
        world_width=800,
        world_height=600,
        agent_count=10,
        steps_per_generation=100,
        max_generations=5
    )
    
    sim = Simulation(config=config)
    
    # Iniciar
    sim.start()
    
    # Simular con datos ficticios
    detected_objects = [(400, 300), (200, 200), (600, 400)]
    
    print("\n🔄 Ejecutando simulación...")
    
    step_count = 0
    while not sim.is_done() and step_count < 300:
        # Variar nivel de alerta
        alert = np.sin(step_count / 50) * 0.5 + 0.5
        
        sim.update(detected_objects, alert_level=alert)
        step_count += 1
    
    # Estadísticas finales
    stats = sim.get_statistics()
    print(f"\n📊 Estadísticas finales:")
    print(f"   Generación: {stats['generation']}")
    print(f"   Pasos totales: {stats['total_steps']}")
    print(f"   Mejor fitness: {stats.get('best_fitness', 0):.2f}")
    
    sim.stop()
    
    print("\n✅ Test completado exitosamente!")
