"""
Red neuronal para el cerebro de los agentes.

Este módulo implementa una red neuronal feedforward simple
que sirve como "cerebro" de cada agente del ecosistema.
Los pesos de la red se optimizan mediante algoritmos genéticos.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NeuralBrainConfig:
    """Configuración de la red neuronal del agente."""
    input_size: int = 8      # Entradas (sensores)
    hidden_size: int = 16    # Neuronas ocultas
    output_size: int = 4     # Salidas (acciones)
    activation: str = "tanh" # Función de activación


class NeuralBrain:
    """
    Red neuronal feedforward para agentes evolutivos.
    
    Esta red neuronal simple tiene una capa oculta y se utiliza
    para tomar decisiones en el ecosistema. Los pesos no se
    entrenan con backpropagation, sino que se optimizan
    mediante algoritmos genéticos.
    
    Arquitectura:
        Input (8) → Hidden (16) → Output (4)
    
    Attributes:
        config: Configuración de la red
        weights1: Pesos de entrada a oculta
        bias1: Bias de capa oculta
        weights2: Pesos de oculta a salida
        bias2: Bias de capa de salida
    
    Example:
        >>> brain = NeuralBrain()
        >>> inputs = np.array([0.5, 0.3, 0.8, 0.1, 0.2, 0.4, 0.6, 0.9])
        >>> outputs = brain.forward(inputs)
        >>> action = np.argmax(outputs)  # Elegir mejor acción
    """
    
    def __init__(
        self,
        config: Optional[NeuralBrainConfig] = None,
        genome: Optional[np.ndarray] = None
    ):
        """
        Inicializa la red neuronal.
        
        Args:
            config: Configuración de la red
            genome: Vector de pesos (opcional, si no se pasa se inicializa aleatorio)
        """
        self.config = config or NeuralBrainConfig()
        
        # Calcular tamaños
        self._w1_size = self.config.input_size * self.config.hidden_size
        self._b1_size = self.config.hidden_size
        self._w2_size = self.config.hidden_size * self.config.output_size
        self._b2_size = self.config.output_size
        
        self._total_params = self._w1_size + self._b1_size + self._w2_size + self._b2_size
        
        if genome is not None:
            self.set_genome(genome)
        else:
            self._initialize_random()
    
    def _initialize_random(self):
        """Inicializa pesos aleatorios con distribución Xavier."""
        # Xavier initialization para mejor convergencia
        w1_std = np.sqrt(2.0 / (self.config.input_size + self.config.hidden_size))
        w2_std = np.sqrt(2.0 / (self.config.hidden_size + self.config.output_size))
        
        self.weights1 = np.random.randn(
            self.config.input_size, self.config.hidden_size
        ) * w1_std
        
        self.bias1 = np.zeros(self.config.hidden_size)
        
        self.weights2 = np.random.randn(
            self.config.hidden_size, self.config.output_size
        ) * w2_std
        
        self.bias2 = np.zeros(self.config.output_size)
    
    @property
    def genome_size(self) -> int:
        """Tamaño total del genoma (todos los parámetros)."""
        return self._total_params
    
    def get_genome(self) -> np.ndarray:
        """
        Obtiene el genoma (todos los pesos) como un vector 1D.
        
        Returns:
            Vector numpy con todos los parámetros de la red
        """
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten()
        ])
    
    def set_genome(self, genome: np.ndarray):
        """
        Establece los pesos de la red desde un genoma.
        
        Args:
            genome: Vector con todos los parámetros
        """
        if len(genome) != self._total_params:
            raise ValueError(
                f"Genoma debe tener {self._total_params} elementos, "
                f"recibió {len(genome)}"
            )
        
        idx = 0
        
        # Pesos1: input_size x hidden_size
        self.weights1 = genome[idx:idx + self._w1_size].reshape(
            self.config.input_size, self.config.hidden_size
        )
        idx += self._w1_size
        
        # Bias1: hidden_size
        self.bias1 = genome[idx:idx + self._b1_size]
        idx += self._b1_size
        
        # Pesos2: hidden_size x output_size
        self.weights2 = genome[idx:idx + self._w2_size].reshape(
            self.config.hidden_size, self.config.output_size
        )
        idx += self._w2_size
        
        # Bias2: output_size
        self.bias2 = genome[idx:idx + self._b2_size]
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Aplica la función de activación."""
        if self.config.activation == "tanh":
            return np.tanh(x)
        elif self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return np.tanh(x)  # Default
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante.
        
        Args:
            inputs: Vector de entrada (tamaño = input_size)
        
        Returns:
            Vector de salida (tamaño = output_size)
        """
        # Asegurar que la entrada tiene el tamaño correcto
        if len(inputs) != self.config.input_size:
            # Padding o truncamiento
            if len(inputs) < self.config.input_size:
                inputs = np.pad(inputs, (0, self.config.input_size - len(inputs)))
            else:
                inputs = inputs[:self.config.input_size]
        
        # Capa oculta
        hidden = self._activation(np.dot(inputs, self.weights1) + self.bias1)
        
        # Capa de salida
        output = self._activation(np.dot(hidden, self.weights2) + self.bias2)
        
        return output
    
    def copy(self) -> 'NeuralBrain':
        """Crea una copia profunda de la red."""
        new_brain = NeuralBrain(config=self.config)
        new_brain.set_genome(self.get_genome().copy())
        return new_brain
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """
        Muta los pesos de la red in-place.
        
        Args:
            mutation_rate: Probabilidad de mutar cada peso
            mutation_strength: Desviación estándar de la mutación
        """
        genome = self.get_genome()
        
        # Máscara de mutación
        mutation_mask = np.random.random(len(genome)) < mutation_rate
        
        # Aplicar mutaciones gaussianas
        mutations = np.random.randn(len(genome)) * mutation_strength
        genome[mutation_mask] += mutations[mutation_mask]
        
        self.set_genome(genome)
    
    @staticmethod
    def crossover(parent1: 'NeuralBrain', parent2: 'NeuralBrain') -> 'NeuralBrain':
        """
        Cruzamiento de dos cerebros para producir un hijo.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
        
        Returns:
            Nuevo cerebro hijo
        """
        genome1 = parent1.get_genome()
        genome2 = parent2.get_genome()
        
        # Crossover uniforme
        mask = np.random.random(len(genome1)) > 0.5
        child_genome = np.where(mask, genome1, genome2)
        
        child = NeuralBrain(config=parent1.config)
        child.set_genome(child_genome)
        
        return child


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test de la Red Neuronal del Agente")
    print("=" * 50)
    
    # Crear cerebro
    config = NeuralBrainConfig(input_size=8, hidden_size=16, output_size=4)
    brain = NeuralBrain(config=config)
    
    print(f"\n📊 Arquitectura: {config.input_size} → {config.hidden_size} → {config.output_size}")
    print(f"   Total parámetros: {brain.genome_size}")
    
    # Probar forward
    inputs = np.random.randn(8)
    outputs = brain.forward(inputs)
    
    print(f"\n🔄 Forward pass:")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output values: {outputs}")
    
    # Probar copia y mutación
    brain_copy = brain.copy()
    brain_copy.mutate(mutation_rate=0.2)
    
    # Verificar que son diferentes
    diff = np.sum(np.abs(brain.get_genome() - brain_copy.get_genome()))
    print(f"\n🧬 Diferencia después de mutación: {diff:.4f}")
    
    # Probar crossover
    parent2 = NeuralBrain(config=config)
    child = NeuralBrain.crossover(brain, parent2)
    
    print(f"   Crossover exitoso: child genome size = {len(child.get_genome())}")
    
    print("\n✅ Test completado exitosamente!")
