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
    """Configuración avanzada del agente cibernético."""
    max_speed: float = 5.0       # Velocidad máxima
    max_energy: float = 100.0    # Energía máxima
    metabolism: float = 0.1      # Costo de existir por frame
    move_cost: float = 0.2       # Costo por unidad de movimiento
    sensor_range: float = 150.0  # Rango de visión
    size: float = 12.0           # Tamaño visual
    input_size: int = 16         # Sensores aumentados
    output_size: int = 2         # Salidas continuas (Speed, Turn)
    

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
        self.id = agent_id
        self.world_size = world_size
        self.config = config or AgentConfig()
        
        # Estado Bio-Mecánico
        self.position = np.array(position if position else [
            np.random.uniform(50, world_size[0]-50),
            np.random.uniform(50, world_size[1]-50)
        ], dtype=float)
        
        self.velocity = np.array([0.0, 0.0])
        self.angle = np.random.uniform(0, 2 * np.pi)
        
        # Metabolismo
        self.energy = self.config.max_energy
        self.health = 100.0
        self.alive = True
        self.age = 0
        
        # Cerebro (Input 16 -> Hidden -> Output 2)
        if not brain:
            brain_config = NeuralBrainConfig(
                input_size=self.config.input_size,
                output_size=self.config.output_size
            )
            self.brain = NeuralBrain(brain_config)
        else:
            self.brain = brain
            
        self.fitness = 0.0
        self.sensor_data = np.zeros(self.config.input_size)
        
        # Visual
        self._color = self._generate_color()
        self.positions_history = []

    def _generate_color(self) -> Tuple[int, int, int]:
        np.random.seed(self.id)
        # Colores más "robóticos/cyborg" (Azules/Cianes neón)
        base_color = np.random.choice([0, 1, 2]) # R, G, B dominancia
        color = [50, 50, 50]
        color[base_color] = np.random.randint(150, 255)
        color[(base_color + 1) % 3] = np.random.randint(100, 200)
        return tuple(color)

    @property
    def color(self) -> Tuple[int, int, int]:
        """Color del agente para visualización."""
        return self._color

    def perceive(
        self,
        detected_objects: List[Tuple[float, float]], # YOLO Obstacles
        food_items: List[Tuple[float, float]],       # Energy Sources
        alert_level: float = 0.0,
        other_agents: Optional[List['Agent']] = None
    ):
        """
        Sistema sensorial de 16 canales:
        [0-3]  : Olor a Comida (4 cuadrantes)
        [4-7]  : Proximidad a Obstáculos/Humanos (4 cuadrantes)
        [8-11] : Proximidad a Paredes (4 direcciones)
        [12]   : Nivel de Energía Probado
        [13]   : Velocidad Actual
        [14]   : Nivel de Alerta Global
        [15]   : Bias/Memoria (Feedback oscilatorio)
        """
        self.sensor_data.fill(0)
        
        # Función auxiliar para cuadrantes (0: Front, 1: Right, 2: Back, 3: Left)
        def get_quadrant(angle_diff):
            angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
            if -np.pi/4 <= angle_diff < np.pi/4: return 0 # Front
            elif np.pi/4 <= angle_diff < 3*np.pi/4: return 1 # Right
            elif -3*np.pi/4 <= angle_diff < -np.pi/4: return 3 # Left
            else: return 2 # Back

        # 1. Comida (Green dots)
        if food_items:
            for food in food_items:
                vec = np.array(food) - self.position
                dist = np.linalg.norm(vec)
                if dist < self.config.sensor_range:
                    angle_to = np.arctan2(vec[1], vec[0])
                    quad = get_quadrant(angle_to - self.angle)
                    intensity = 1.0 - (dist / self.config.sensor_range)
                    self.sensor_data[quad] = max(self.sensor_data[quad], intensity)

        # 2. Obstáculos Humanos (YOLO)
        if detected_objects:
            for obj in detected_objects:
                vec = np.array(obj) - self.position
                dist = np.linalg.norm(vec)
                if dist < self.config.sensor_range:
                    angle_to = np.arctan2(vec[1], vec[0])
                    quad = get_quadrant(angle_to - self.angle)
                    intensity = 1.0 - (dist / self.config.sensor_range)
                    self.sensor_data[4 + quad] = max(self.sensor_data[4 + quad], intensity)
        
        # 3. Paredes (Front, Right, Back, Left relativas)
        # Simplificación: Proyección de rayos
        # Front Wall
        ray_x = self.position[0] + np.cos(self.angle) * self.config.sensor_range
        ray_y = self.position[1] + np.sin(self.angle) * self.config.sensor_range
        # (Lógica simplificada de distancia a bordes)
        self.sensor_data[8] = min(self.position[0], self.world_size[0]-self.position[0]) / self.world_size[0] # H-Center
        self.sensor_data[9] = min(self.position[1], self.world_size[1]-self.position[1]) / self.world_size[1] # V-Center
        # Usamos sensores 8-11 para posición relativa normalizada en el mapa en lugar de sonar complejo
        self.sensor_data[10] = self.position[0] / self.world_size[0] # X relativo
        self.sensor_data[11] = self.position[1] / self.world_size[1] # Y relativo

        # 4. Propiocepción
        self.sensor_data[12] = self.energy / self.config.max_energy
        self.sensor_data[13] = np.linalg.norm(self.velocity) / self.config.max_speed
        self.sensor_data[14] = alert_level
        self.sensor_data[15] = np.sin(self.age * 0.1) # Oscilador interno (ritmo cardíaco)

    def decide(self):
        """Red Neuronal produce control continuo: Velocidad y Giro."""
        output = self.brain.forward(self.sensor_data)
        
        # Output 0: Aceleración automática (-1 freno, +1 full gas)
        target_speed_pct = np.tanh(output[0]) 
        
        # Output 1: Giro (-1 izquierda, +1 derecha)
        turn_force = np.tanh(output[1]) 
        
        return target_speed_pct, turn_force

    def act(self, action):
        speed_pct, turn_force = action
        
        # 1. Aplicar Giro
        self.angle += turn_force * 0.2
        
        # 2. Calcular Velocidad
        # Mapear -1..1 a 0..max_speed (si speed_pct < 0, frena/retrocede lento)
        speed = speed_pct * self.config.max_speed
        
        self.velocity[0] = np.cos(self.angle) * speed
        self.velocity[1] = np.sin(self.angle) * speed
        
        # 3. Consumo Metabólico (Costo de existir + Costo de moverse)
        energy_loss = self.config.metabolism + (abs(speed) * self.config.move_cost)
        self.energy -= energy_loss
        
        # 4. Muerte
        if self.energy <= 0:
            self.alive = False
            self.energy = 0

    def update(self):
        if not self.alive: return

        self.position += self.velocity
        self.age += 1
        
        # Límites del mundo (Bounce suave)
        margin = 5
        if self.position[0] < margin: 
            self.position[0] = margin
            self.angle = np.pi - self.angle
        elif self.position[0] > self.world_size[0]-margin: 
            self.position[0] = self.world_size[0]-margin
            self.angle = np.pi - self.angle
            
        if self.position[1] < margin: 
            self.position[1] = margin
            self.angle = -self.angle
        elif self.position[1] > self.world_size[1]-margin: 
            self.position[1] = self.world_size[1]-margin
            self.angle = -self.angle
            
        self.angle %= (2*np.pi)
        
        # Historial corto
        self.positions_history.append(tuple(self.position))
        if len(self.positions_history) > 50:
            self.positions_history.pop(0)

    def eat(self):
        """Recuperar energía al comer."""
        self.energy = min(self.energy + 30.0, self.config.max_energy)
        self.fitness += 10.0 # Recompensa evolutiva directa

    def copy(self) -> 'Agent':
        new_agent = Agent(
            agent_id=self.id,
            world_size=self.world_size,
            config=self.config,
            brain=self.brain.copy()
        )
        return new_agent
        
    def calculate_fitness(self, detected_objects, alert_level):
        """
        Fitness v2: Supervivencia + Energía.
        No necesitamos lógica compleja aquí, el fitness se acumula en 'eat()'
        y simplemente por sobrevivir cada frame.
        """
        if self.alive:
            self.fitness += 0.1 # Recompensa por vivir un frame más
            
    def reset(self):
        """Resurrección para nueva generación."""
        self.energy = self.config.max_energy
        self.alive = True
        self.age = 0
        self.fitness = 0.0
        self.position = np.array([
            np.random.uniform(50, self.world_size[0]-50),
            np.random.uniform(50, self.world_size[1]-50)
        ])
        self.positions_history.clear()


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
