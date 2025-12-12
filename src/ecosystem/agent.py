"""
Agente del ecosistema con red neuronal como cerebro.

Cada agente tiene:
- Posición en el mundo 2D
- Cerebro (red neuronal) que decide acciones
- Sensores para percibir el entorno
- Fitness que mide su desempeño
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .neural_brain import NeuralBrain, NeuralBrainConfig


@dataclass
class AgentConfig:
    """Configuración del agente."""
    speed: float = 3.0           # Velocidad de movimiento
    sensor_range: float = 100.0  # Rango de sensores
    size: float = 10.0           # Tamaño visual
    

class Agent:
    """
    Agente virtual con cerebro neuronal.
    
    Cada agente puede:
    - Percibir su entorno (objetos detectados, bordes, otros agentes)
    - Decidir acciones usando su red neuronal
    - Moverse en el mundo 2D
    - Acumular fitness basado en su desempeño
    
    Attributes:
        id: Identificador único
        position: Posición (x, y) en el mundo
        velocity: Velocidad actual (vx, vy)
        brain: Red neuronal que controla al agente
        fitness: Puntuación acumulada
        alive: Estado del agente
    
    Example:
        >>> agent = Agent(agent_id=1, world_size=(800, 600))
        >>> agent.perceive(detected_objects, other_agents)
        >>> agent.decide()
        >>> agent.update()
    """
    
    # Acciones posibles
    ACTION_FORWARD = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_STOP = 3
    
    def __init__(
        self,
        agent_id: int,
        world_size: Tuple[int, int],
        position: Optional[Tuple[float, float]] = None,
        config: Optional[AgentConfig] = None,
        brain: Optional[NeuralBrain] = None
    ):
        """
        Inicializa un agente.
        
        Args:
            agent_id: ID único del agente
            world_size: Tamaño del mundo (width, height)
            position: Posición inicial (opcional, aleatorio si no se da)
            config: Configuración del agente
            brain: Cerebro del agente (opcional, nuevo aleatorio si no se da)
        """
        self.id = agent_id
        self.world_size = world_size
        self.config = config or AgentConfig()
        
        # Posición inicial
        if position:
            self.position = np.array(position, dtype=float)
        else:
            self.position = np.array([
                np.random.uniform(50, world_size[0] - 50),
                np.random.uniform(50, world_size[1] - 50)
            ])
        
        # Velocidad y dirección
        self.velocity = np.array([0.0, 0.0])
        self.angle = np.random.uniform(0, 2 * np.pi)
        
        # Cerebro
        self.brain = brain or NeuralBrain(NeuralBrainConfig())
        
        # Estado
        self.alive = True
        self.fitness = 0.0
        self.age = 0
        
        # Sensores
        self.sensor_data = np.zeros(8)
        
        # Historial
        self.positions_history: List[Tuple[float, float]] = []
        self.detections_count = 0
        
        # Color (basado en ID para visualización)
        self._color = self._generate_color()
    
    def _generate_color(self) -> Tuple[int, int, int]:
        """Genera un color único basado en el ID."""
        np.random.seed(self.id)
        color = tuple(np.random.randint(100, 255, 3).tolist())
        np.random.seed(None)  # Reset seed
        return color
    
    @property
    def color(self) -> Tuple[int, int, int]:
        """Color del agente para visualización."""
        return self._color
    
    def perceive(
        self,
        detected_objects: List[Tuple[float, float]],
        alert_level: float = 0.0,
        other_agents: Optional[List['Agent']] = None
    ):
        """
        Percibe el entorno y actualiza los sensores.
        
        Los 8 sensores capturan:
        - [0-3]: Distancia a objetos detectados en 4 direcciones
        - [4]: Nivel de alerta actual
        - [5]: Distancia al borde más cercano
        - [6]: Densidad de agentes cercanos
        - [7]: Ángulo hacia el objeto más cercano
        
        Args:
            detected_objects: Lista de posiciones (x, y) de objetos detectados
            alert_level: Nivel de alerta del sistema difuso (0-1)
            other_agents: Lista de otros agentes en el mundo
        """
        # Resetear sensores
        self.sensor_data = np.zeros(8)
        
        # Sensor 0-3: Distancia a objetos en 4 direcciones
        if detected_objects:
            for i, direction in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
                min_dist = self.config.sensor_range
                for obj_pos in detected_objects:
                    obj_vec = np.array(obj_pos) - self.position
                    dist = np.linalg.norm(obj_vec)
                    if dist < min_dist and dist < self.config.sensor_range:
                        # Verificar si está en la dirección correcta
                        angle_to_obj = np.arctan2(obj_vec[1], obj_vec[0])
                        angle_diff = abs(angle_to_obj - (self.angle + direction))
                        if angle_diff < np.pi / 4:  # Cono de 45 grados
                            min_dist = dist
                
                # Normalizar distancia (1 = cerca, 0 = lejos)
                self.sensor_data[i] = 1.0 - (min_dist / self.config.sensor_range)
        
        # Sensor 4: Nivel de alerta
        self.sensor_data[4] = alert_level
        
        # Sensor 5: Distancia al borde más cercano (normalizada)
        border_dists = [
            self.position[0],  # Izquierda
            self.world_size[0] - self.position[0],  # Derecha
            self.position[1],  # Arriba
            self.world_size[1] - self.position[1]  # Abajo
        ]
        min_border = min(border_dists)
        self.sensor_data[5] = 1.0 - min(min_border / 100, 1.0)
        
        # Sensor 6: Densidad de agentes cercanos
        if other_agents:
            nearby = sum(
                1 for agent in other_agents
                if agent.id != self.id and
                np.linalg.norm(agent.position - self.position) < self.config.sensor_range
            )
            self.sensor_data[6] = min(nearby / 5, 1.0)  # Normalizar (max 5 cercanos = 1.0)
        
        # Sensor 7: Ángulo hacia objeto más cercano (normalizado)
        if detected_objects:
            closest = min(detected_objects, 
                         key=lambda p: np.linalg.norm(np.array(p) - self.position))
            vec_to_closest = np.array(closest) - self.position
            angle_to_closest = np.arctan2(vec_to_closest[1], vec_to_closest[0])
            angle_diff = angle_to_closest - self.angle
            # Normalizar a [-1, 1]
            self.sensor_data[7] = angle_diff / np.pi
    
    def decide(self) -> int:
        """
        Usa el cerebro para decidir la siguiente acción.
        
        Returns:
            Índice de la acción elegida (0-3)
        """
        # Obtener salida de la red neuronal
        outputs = self.brain.forward(self.sensor_data)
        
        # Elegir acción con mayor valor
        action = np.argmax(outputs)
        
        return action
    
    def act(self, action: int):
        """
        Ejecuta una acción.
        
        Args:
            action: Índice de la acción (0-3)
        """
        if action == self.ACTION_FORWARD:
            # Moverse hacia adelante
            self.velocity[0] = np.cos(self.angle) * self.config.speed
            self.velocity[1] = np.sin(self.angle) * self.config.speed
        
        elif action == self.ACTION_LEFT:
            # Girar a la izquierda
            self.angle -= 0.2
            self.velocity[0] = np.cos(self.angle) * self.config.speed * 0.5
            self.velocity[1] = np.sin(self.angle) * self.config.speed * 0.5
        
        elif action == self.ACTION_RIGHT:
            # Girar a la derecha
            self.angle += 0.2
            self.velocity[0] = np.cos(self.angle) * self.config.speed * 0.5
            self.velocity[1] = np.sin(self.angle) * self.config.speed * 0.5
        
        elif action == self.ACTION_STOP:
            # Quedarse quieto
            self.velocity *= 0.5
    
    def update(self):
        """Actualiza la posición del agente."""
        # Actualizar posición
        self.position += self.velocity
        
        # Mantener dentro del mundo (bounce)
        if self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] *= -0.5
            self.angle = np.pi - self.angle
        elif self.position[0] > self.world_size[0]:
            self.position[0] = self.world_size[0]
            self.velocity[0] *= -0.5
            self.angle = np.pi - self.angle
        
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] *= -0.5
            self.angle = -self.angle
        elif self.position[1] > self.world_size[1]:
            self.position[1] = self.world_size[1]
            self.velocity[1] *= -0.5
            self.angle = -self.angle
        
        # Normalizar ángulo
        self.angle = self.angle % (2 * np.pi)
        
        # Incrementar edad
        self.age += 1
        
        # Guardar historial (últimas 100 posiciones)
        self.positions_history.append(tuple(self.position))
        if len(self.positions_history) > 100:
            self.positions_history.pop(0)
    
    def calculate_fitness(
        self,
        detected_objects: List[Tuple[float, float]],
        alert_level: float
    ):
        """
        Calcula y actualiza el fitness del agente.
        
        El fitness recompensa:
        - Estar cerca de objetos detectados cuando hay alerta alta
        - Moverse y explorar el mundo
        - Evitar chocar con los bordes
        
        Args:
            detected_objects: Objetos detectados
            alert_level: Nivel de alerta actual
        """
        fitness_delta = 0.0
        
        # Recompensa por estar cerca de objetos detectados
        if detected_objects and alert_level > 0.3:
            closest_dist = min(
                np.linalg.norm(np.array(p) - self.position)
                for p in detected_objects
            )
            
            if closest_dist < self.config.sensor_range:
                # Más cerca = más recompensa
                proximity_reward = (1.0 - closest_dist / self.config.sensor_range)
                fitness_delta += proximity_reward * alert_level * 2.0
                self.detections_count += 1
        
        # Recompensa por moverse (exploración)
        speed = np.linalg.norm(self.velocity)
        fitness_delta += speed * 0.01
        
        # Penalización por estar muy cerca del borde
        border_dists = [
            self.position[0],
            self.world_size[0] - self.position[0],
            self.position[1],
            self.world_size[1] - self.position[1]
        ]
        if min(border_dists) < 20:
            fitness_delta -= 0.1
        
        # Actualizar fitness total
        self.fitness += fitness_delta
    
    def reset(self):
        """Reinicia el agente a su estado inicial."""
        self.position = np.array([
            np.random.uniform(50, self.world_size[0] - 50),
            np.random.uniform(50, self.world_size[1] - 50)
        ])
        self.velocity = np.array([0.0, 0.0])
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.fitness = 0.0
        self.age = 0
        self.alive = True
        self.positions_history.clear()
        self.detections_count = 0
    
    def copy(self) -> 'Agent':
        """Crea una copia del agente con el mismo cerebro."""
        new_agent = Agent(
            agent_id=self.id,
            world_size=self.world_size,
            config=self.config,
            brain=self.brain.copy()
        )
        return new_agent


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test del Agente")
    print("=" * 50)
    
    # Crear agente
    agent = Agent(agent_id=1, world_size=(800, 600))
    
    print(f"\n📍 Agente creado:")
    print(f"   Posición: {agent.position}")
    print(f"   Ángulo: {agent.angle:.2f} rad")
    print(f"   Fitness: {agent.fitness}")
    
    # Simular algunos pasos
    detected_objects = [(400, 300), (200, 200)]
    
    for step in range(10):
        agent.perceive(detected_objects, alert_level=0.5)
        action = agent.decide()
        agent.act(action)
        agent.update()
        agent.calculate_fitness(detected_objects, alert_level=0.5)
    
    print(f"\n📊 Después de 10 pasos:")
    print(f"   Posición: {agent.position}")
    print(f"   Fitness: {agent.fitness:.2f}")
    print(f"   Edad: {agent.age}")
    
    # Probar copia
    agent_copy = agent.copy()
    print(f"\n🧬 Copia creada con fitness: {agent_copy.fitness}")
    
    print("\n✅ Test completado exitosamente!")
