"""
Algoritmo Genético para evolucionar agentes.

Este módulo implementa el algoritmo genético que optimiza
los cerebros de los agentes a través de selección, cruzamiento
y mutación.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .agent import Agent
from .neural_brain import NeuralBrain


@dataclass
class GeneticConfig:
    """Configuración del algoritmo genético."""
    population_size: int = 20       # Tamaño de la población
    mutation_rate: float = 0.1      # Probabilidad de mutación
    mutation_strength: float = 0.5  # Fuerza de la mutación
    crossover_rate: float = 0.7     # Probabilidad de cruzamiento
    elitism: int = 2                # Número de élites preservados
    tournament_size: int = 3        # Tamaño del torneo


class GeneticAlgorithm:
    """
    Algoritmo genético para evolucionar poblaciones de agentes.
    
    El algoritmo sigue estos pasos:
    1. Evaluar fitness de todos los agentes
    2. Seleccionar los mejores (elitismo + torneo)
    3. Cruzar para crear nuevos individuos
    4. Mutar la nueva generación
    5. Reemplazar la población
    
    Attributes:
        config: Configuración del algoritmo
        generation: Número de generación actual
        best_fitness_history: Historial del mejor fitness
        avg_fitness_history: Historial del fitness promedio
    
    Example:
        >>> ga = GeneticAlgorithm()
        >>> agents = ga.create_population(world_size=(800, 600))
        >>> # ... simular y calcular fitness ...
        >>> new_agents = ga.evolve(agents)
    """
    
    def __init__(self, config: Optional[GeneticConfig] = None):
        """
        Inicializa el algoritmo genético.
        
        Args:
            config: Configuración del algoritmo
        """
        self.config = config or GeneticConfig()
        self.generation = 0
        self.stagnation_counter = 0     # Contador de estancamiento
        self.current_mutation_rate = self.config.mutation_rate
        self.current_mutation_strength = self.config.mutation_strength
        
        # Historial
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.worst_fitness_history: List[float] = []
        
        print(f"🧬 Algoritmo Genético inicializado:")
        print(f"   Población: {self.config.population_size}")
        print(f"   Mutación: {self.config.mutation_rate * 100:.0f}%")
        print(f"   Elitismo: {self.config.elitism}")
    
    def create_population(
        self,
        world_size: Tuple[int, int]
    ) -> List[Agent]:
        """
        Crea una población inicial de agentes.
        
        Args:
            world_size: Tamaño del mundo (width, height)
        
        Returns:
            Lista de agentes con cerebros aleatorios
        """
        population = []
        
        for i in range(self.config.population_size):
            agent = Agent(agent_id=i, world_size=world_size)
            population.append(agent)
        
        print(f"✅ Población inicial creada: {len(population)} agentes")
        return population
    
    def select_parent(self, population: List[Agent]) -> Agent:
        """
        Selecciona un padre usando selección por torneo.
        
        Args:
            population: Lista de agentes
        
        Returns:
            Agente seleccionado como padre
        """
        # Seleccionar participantes aleatorios del torneo
        tournament = np.random.choice(
            population,
            size=min(self.config.tournament_size, len(population)),
            replace=False
        )
        
        # El ganador es el de mayor fitness
        winner = max(tournament, key=lambda a: a.fitness)
        return winner
    
    def crossover(self, parent1: Agent, parent2: Agent, child_id: int) -> Agent:
        """
        Realiza cruzamiento entre dos padres.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            child_id: ID para el hijo
        
        Returns:
            Nuevo agente hijo
        """
        # Crear nuevo cerebro por cruzamiento
        child_brain = NeuralBrain.crossover(parent1.brain, parent2.brain)
        
        # Crear nuevo agente con el cerebro hijo
        child = Agent(
            agent_id=child_id,
            world_size=parent1.world_size,
            config=parent1.config,
            brain=child_brain
        )
        
        return child
    
    def mutate(self, agent: Agent):
        """
        Muta el cerebro de un agente in-place.
        
        Args:
            agent: Agente a mutar
        """
        agent.brain.mutate(
            mutation_rate=self.current_mutation_rate,
            mutation_strength=self.current_mutation_strength
        )
    
    def _adjust_mutation_rate(self):
        """Ajusta la tasa de mutación basado en el progreso."""
        if len(self.avg_fitness_history) < 2:
            return

        current_avg = self.avg_fitness_history[-1]
        prev_avg = self.avg_fitness_history[-2]
        
        # Si la mejora es mínima (< 1%), incrementar estancamiento
        if (current_avg - prev_avg) < (prev_avg * 0.01):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0 # Reset si hay mejora
            
        # Lógica adaptativa
        if self.stagnation_counter > 5:
            # Crisis: Aumentar drásticamente mutación para escapar del mínimo local
            self.current_mutation_rate = min(0.5, self.config.mutation_rate * 3.0)
            self.current_mutation_strength = self.config.mutation_strength * 2.0
            print(f"⚠️ Estancamiento detectado ({self.stagnation_counter} gen). Mutación boost: {self.current_mutation_rate:.2f}")
        else:
            # Normalidad: Restaurar valores base
            self.current_mutation_rate = self.config.mutation_rate
            self.current_mutation_strength = self.config.mutation_strength
    
    def evolve(self, population: List[Agent]) -> List[Agent]:
        """
        Evoluciona la población a la siguiente generación.
        
        Args:
            population: Población actual
        
        Returns:
            Nueva población evolucionada
        """
        # Ordenar por fitness (mayor a menor)
        sorted_pop = sorted(population, key=lambda a: a.fitness, reverse=True)
        
        # Registrar estadísticas
        fitnesses = [a.fitness for a in population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))
        self.worst_fitness_history.append(min(fitnesses))
        
        # Ajustar mutación dinámicamente
        self._adjust_mutation_rate()
        
        new_population = []
        
        # Elitismo: copiar los mejores sin cambios
        for i in range(self.config.elitism):
            elite = sorted_pop[i].copy()
            elite.id = i
            elite.reset()  # Resetear estado pero mantener cerebro
            new_population.append(elite)
        
        # Generar el resto de la población
        while len(new_population) < self.config.population_size:
            # Seleccionar padres
            parent1 = self.select_parent(sorted_pop)
            parent2 = self.select_parent(sorted_pop)
            
            # Asegurar que son diferentes
            while parent2.id == parent1.id and len(sorted_pop) > 1:
                parent2 = self.select_parent(sorted_pop)
            
            # Cruzamiento
            if np.random.random() < self.config.crossover_rate:
                child = self.crossover(parent1, parent2, len(new_population))
            else:
                # Sin cruzamiento, copiar al mejor padre
                child = parent1.copy()
                child.id = len(new_population)
            
            # Mutación
            self.mutate(child)
            
            # Resetear estado
            child.reset()
            
            new_population.append(child)
        
        self.generation += 1
        
        return new_population
    
    def get_best_agent(self, population: List[Agent]) -> Agent:
        """Retorna el agente con mayor fitness."""
        return max(population, key=lambda a: a.fitness)
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas de la evolución.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.best_fitness_history:
            return {
                "generation": self.generation,
                "best_fitness": 0.0,
                "avg_fitness": 0.0,
                "improvement": 0.0
            }
        
        best = self.best_fitness_history[-1]
        avg = self.avg_fitness_history[-1]
        
        # Calcular mejora respecto a la primera generación
        if len(self.best_fitness_history) > 1:
            improvement = best - self.best_fitness_history[0]
        else:
            improvement = 0.0
        
        return {
            "generation": self.generation,
            "best_fitness": best,
            "avg_fitness": avg,
            "improvement": improvement,
            "best_history": self.best_fitness_history,
            "avg_history": self.avg_fitness_history
        }
    
    def save_best_genome(self, population: List[Agent], filepath: str):
        """
        Guarda el genoma del mejor agente.
        
        Args:
            population: Población actual
            filepath: Ruta del archivo
        """
        best = self.get_best_agent(population)
        genome = best.brain.get_genome()
        np.save(filepath, genome)
        print(f"✅ Mejor genoma guardado en: {filepath}")
    
    def load_genome(self, filepath: str) -> np.ndarray:
        """
        Carga un genoma desde archivo.
        
        Args:
            filepath: Ruta del archivo
        
        Returns:
            Genoma cargado
        """
        genome = np.load(filepath)
        print(f"✅ Genoma cargado desde: {filepath}")
        return genome


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test del Algoritmo Genético")
    print("=" * 50)
    
    # Crear algoritmo
    config = GeneticConfig(population_size=10, mutation_rate=0.1, elitism=2)
    ga = GeneticAlgorithm(config=config)
    
    # Crear población
    population = ga.create_population(world_size=(800, 600))
    
    # Simular algunas generaciones
    for gen in range(5):
        # Asignar fitness aleatorio para prueba
        for agent in population:
            agent.fitness = np.random.uniform(0, 100)
        
        # Obtener mejor antes de evolucionar
        best = ga.get_best_agent(population)
        print(f"\n🌿 Generación {gen + 1}:")
        print(f"   Mejor fitness: {best.fitness:.2f}")
        print(f"   Promedio: {np.mean([a.fitness for a in population]):.2f}")
        
        # Evolucionar
        population = ga.evolve(population)
    
    # Estadísticas finales
    stats = ga.get_statistics()
    print(f"\n📊 Estadísticas finales:")
    print(f"   Generaciones: {stats['generation']}")
    print(f"   Mejor fitness: {stats['best_fitness']:.2f}")
    
    print("\n✅ Test completado exitosamente!")
