# Módulo Ecosystem
"""
Ecosistema de agentes evolutivos con Algoritmos Genéticos y Redes Neuronales.

Este módulo proporciona:
- Agent: Agente virtual con cerebro neural
- NeuralBrain: Red neuronal para toma de decisiones
- GeneticAlgorithm: Algoritmo genético para evolución
- Simulation: Simulación completa del ecosistema

Flujo:
1. Se crea una población de agentes con cerebros aleatorios
2. Los agentes perciben el entorno, deciden y actúan
3. Se evalúa el fitness de cada agente
4. El algoritmo genético evoluciona la población
5. Repetir con la nueva generación

Example:
    >>> from src.ecosystem import Simulation
    >>> 
    >>> sim = Simulation()
    >>> sim.start()
    >>> while not sim.is_done():
    ...     sim.update(detected_objects, alert_level)
"""

from .neural_brain import NeuralBrain, NeuralBrainConfig
from .agent import Agent, AgentConfig
from .genetics import GeneticAlgorithm, GeneticConfig
from .simulation import Simulation, SimulationConfig

__all__ = [
    "NeuralBrain",
    "NeuralBrainConfig",
    "Agent",
    "AgentConfig",
    "GeneticAlgorithm",
    "GeneticConfig",
    "Simulation",
    "SimulationConfig"
]
