"""
Renderizador visual del ecosistema usando Pygame.

Este módulo se encarga de dibujar:
- El mundo 2D con los agentes
- Las detecciones del sistema
- Indicadores de alerta
- Estadísticas de evolución
"""

import pygame
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuración del renderizado."""
    width: int = 800
    height: int = 600
    fps: int = 30
    background_color: Tuple[int, int, int] = (20, 20, 30)
    grid_color: Tuple[int, int, int] = (40, 40, 50)
    agent_size: int = 8
    detection_color: Tuple[int, int, int] = (255, 100, 100)
    show_trails: bool = True
    trail_length: int = 20


class EcosystemRenderer:
    """
    Renderizador del ecosistema en 2D.
    
    Esta clase dibuja los agentes, detecciones y estadísticas
    en una superficie de Pygame.
    
    Attributes:
        config: Configuración del renderizado
        surface: Superficie de Pygame para dibujar
    """
    
    # Colores predefinidos para alertas
    ALERT_COLORS = {
        "normal": (0, 255, 100),
        "precaución": (255, 255, 0),
        "alerta": (255, 165, 0),
        "emergencia": (255, 0, 0)
    }
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Inicializa el renderizador.
        
        Args:
            config: Configuración del renderizado
        """
        self.config = config or RenderConfig()
        self.surface: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
    
    def initialize(self, surface: pygame.Surface):
        """
        Inicializa con una superficie de Pygame.
        
        Args:
            surface: Superficie donde dibujar
        """
        self.surface = surface
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.small_font = pygame.font.SysFont('Arial', 12)
    
    def clear(self):
        """Limpia la superficie con el color de fondo."""
        if self.surface:
            self.surface.fill(self.config.background_color)
            self._draw_grid()
    
    def _draw_grid(self):
        """Dibuja una cuadrícula de fondo."""
        if not self.surface:
            return
        
        grid_spacing = 50
        
        # Líneas verticales
        for x in range(0, self.config.width, grid_spacing):
            pygame.draw.line(
                self.surface,
                self.config.grid_color,
                (x, 0),
                (x, self.config.height),
                1
            )
        
        # Líneas horizontales
        for y in range(0, self.config.height, grid_spacing):
            pygame.draw.line(
                self.surface,
                self.config.grid_color,
                (0, y),
                (self.config.width, y),
                1
            )
    
    def draw_agents(
        self,
        agents: List,
        show_trails: bool = True
    ):
        """
        Dibuja todos los agentes.
        
        Args:
            agents: Lista de objetos Agent
            show_trails: Si dibujar rastros de movimiento
        """
        if not self.surface:
            return
        
        for agent in agents:
            pos = agent.position
            color = agent.color
            
            # Dibujar rastro
            if show_trails and len(agent.positions_history) > 1:
                trail = agent.positions_history[-self.config.trail_length:]
                if len(trail) > 1:
                    # Dibujar líneas con opacidad decreciente
                    for i in range(len(trail) - 1):
                        alpha = int(100 * (i / len(trail)))
                        trail_color = (
                            min(color[0], alpha + 30),
                            min(color[1], alpha + 30),
                            min(color[2], alpha + 30)
                        )
                        pygame.draw.line(
                            self.surface,
                            trail_color,
                            (int(trail[i][0]), int(trail[i][1])),
                            (int(trail[i+1][0]), int(trail[i+1][1])),
                            1
                        )
            
            # Dibujar agente
            x, y = int(pos[0]), int(pos[1])
            
            # Cuerpo del agente
            pygame.draw.circle(
                self.surface,
                color,
                (x, y),
                self.config.agent_size
            )
            
            # Indicador de dirección
            angle = agent.angle
            dx = int(np.cos(angle) * self.config.agent_size * 1.5)
            dy = int(np.sin(angle) * self.config.agent_size * 1.5)
            pygame.draw.line(
                self.surface,
                (255, 255, 255),
                (x, y),
                (x + dx, y + dy),
                2
            )
    
    def draw_detections(
        self,
        detections: List[Tuple[float, float]],
        scale_x: float = 1.0,
        scale_y: float = 1.0
    ):
        """
        Dibuja las detecciones como puntos de interés.
        
        Args:
            detections: Lista de posiciones (x, y)
            scale_x: Factor de escala X
            scale_y: Factor de escala Y
        """
        if not self.surface:
            return
        
        for det in detections:
            x = int(det[0] * scale_x)
            y = int(det[1] * scale_y)
            
            # Círculo exterior pulsante
            radius = 15 + int(5 * np.sin(pygame.time.get_ticks() / 200))
            pygame.draw.circle(
                self.surface,
                self.config.detection_color,
                (x, y),
                radius,
                2
            )
            
            # Círculo interior
            pygame.draw.circle(
                self.surface,
                self.config.detection_color,
                (x, y),
                5
            )
    
    def draw_alert_indicator(
        self,
        alert_level: float,
        alert_category: str,
        position: Tuple[int, int] = (10, 10)
    ):
        """
        Dibuja el indicador de nivel de alerta.
        
        Args:
            alert_level: Nivel de alerta (0.0 - 1.0)
            alert_category: Categoría de alerta
            position: Posición del indicador
        """
        if not self.surface or not self.font:
            return
        
        x, y = position
        bar_width = 150
        bar_height = 20
        
        # Fondo del indicador
        pygame.draw.rect(
            self.surface,
            (50, 50, 60),
            (x, y, bar_width + 10, 60),
            border_radius=5
        )
        
        # Etiqueta
        label = self.font.render("Nivel de Alerta", True, (200, 200, 200))
        self.surface.blit(label, (x + 5, y + 2))
        
        # Barra de progreso
        pygame.draw.rect(
            self.surface,
            (40, 40, 50),
            (x + 5, y + 22, bar_width, bar_height),
            border_radius=3
        )
        
        # Relleno de la barra
        fill_width = int(bar_width * alert_level)
        color = self.ALERT_COLORS.get(alert_category, (128, 128, 128))
        if fill_width > 0:
            pygame.draw.rect(
                self.surface,
                color,
                (x + 5, y + 22, fill_width, bar_height),
                border_radius=3
            )
        
        # Texto de categoría
        cat_text = self.font.render(
            f"{alert_category.upper()} ({alert_level:.0%})",
            True,
            color
        )
        self.surface.blit(cat_text, (x + 5, y + 44))
    
    def draw_statistics(
        self,
        stats: dict,
        position: Tuple[int, int] = (10, 80)
    ):
        """
        Dibuja las estadísticas de la simulación.
        
        Args:
            stats: Diccionario con estadísticas
            position: Posición del panel
        """
        if not self.surface or not self.font:
            return
        
        x, y = position
        
        # Fondo del panel
        pygame.draw.rect(
            self.surface,
            (50, 50, 60),
            (x, y, 160, 100),
            border_radius=5
        )
        
        # Título
        title = self.font.render("Estadísticas", True, (200, 200, 200))
        self.surface.blit(title, (x + 5, y + 2))
        
        # Datos
        lines = [
            f"Generación: {stats.get('generation', 0)}",
            f"Paso: {stats.get('step', 0)}/{stats.get('steps_per_gen', 0)}",
            f"Mejor Fitness: {stats.get('best_fitness', 0):.1f}",
            f"Agentes: {stats.get('agent_count', 0)}"
        ]
        
        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, (180, 180, 180))
            self.surface.blit(text, (x + 5, y + 22 + i * 18))
    
    def draw_fitness_graph(
        self,
        best_history: List[float],
        avg_history: List[float],
        position: Tuple[int, int] = (10, 190),
        size: Tuple[int, int] = (160, 80)
    ):
        """
        Dibuja un gráfico de la evolución del fitness.
        
        Args:
            best_history: Historial del mejor fitness
            avg_history: Historial del fitness promedio
            position: Posición del gráfico
            size: Tamaño del gráfico
        """
        if not self.surface or not self.font:
            return
        
        x, y = position
        width, height = size
        
        # Fondo
        pygame.draw.rect(
            self.surface,
            (50, 50, 60),
            (x, y, width, height + 20),
            border_radius=5
        )
        
        # Título
        title = self.small_font.render("Evolución Fitness", True, (200, 200, 200))
        self.surface.blit(title, (x + 5, y + 2))
        
        # Área del gráfico
        graph_x = x + 5
        graph_y = y + 18
        graph_w = width - 10
        graph_h = height - 5
        
        pygame.draw.rect(
            self.surface,
            (30, 30, 40),
            (graph_x, graph_y, graph_w, graph_h)
        )
        
        # Dibujar líneas si hay datos
        if len(best_history) > 1:
            max_val = max(max(best_history), max(avg_history) if avg_history else 1)
            if max_val == 0:
                max_val = 1
            
            # Normalizar y dibujar mejor fitness
            points_best = []
            for i, val in enumerate(best_history[-50:]):  # Últimos 50 puntos
                px = graph_x + int((i / max(len(best_history[-50:]) - 1, 1)) * graph_w)
                py = graph_y + graph_h - int((val / max_val) * graph_h)
                points_best.append((px, py))
            
            if len(points_best) > 1:
                pygame.draw.lines(self.surface, (0, 255, 100), False, points_best, 2)
            
            # Normalizar y dibujar fitness promedio
            if avg_history:
                points_avg = []
                for i, val in enumerate(avg_history[-50:]):
                    px = graph_x + int((i / max(len(avg_history[-50:]) - 1, 1)) * graph_w)
                    py = graph_y + graph_h - int((val / max_val) * graph_h)
                    points_avg.append((px, py))
                
                if len(points_avg) > 1:
                    pygame.draw.lines(self.surface, (100, 100, 255), False, points_avg, 1)
    
    def draw_controls_help(
        self,
        position: Tuple[int, int] = (10, 520)
    ):
        """Dibuja ayuda de controles."""
        if not self.surface or not self.small_font:
            return
        
        x, y = position
        
        controls = [
            "SPACE: Pausar/Reanudar",
            "R: Reiniciar",
            "ESC: Salir"
        ]
        
        for i, ctrl in enumerate(controls):
            text = self.small_font.render(ctrl, True, (120, 120, 120))
            self.surface.blit(text, (x, y + i * 14))


class VideoRenderer:
    """
    Renderizador de video con detecciones.
    
    Convierte frames de OpenCV a superficie de Pygame
    y dibuja las detecciones encima.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (400, 300)):
        """
        Inicializa el renderizador de video.
        
        Args:
            target_size: Tamaño objetivo para el video
        """
        self.target_size = target_size
        self.font = None
    
    def initialize(self):
        """Inicializa las fuentes."""
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 12)
    
    def frame_to_surface(self, frame) -> pygame.Surface:
        """
        Convierte un frame de OpenCV a superficie de Pygame.
        
        Args:
            frame: Frame de OpenCV (numpy array BGR)
        
        Returns:
            Superficie de Pygame
        """
        import cv2
        
        # Redimensionar
        frame = cv2.resize(frame, self.target_size)
        
        # Convertir BGR a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Rotar y voltear para Pygame
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        
        # Crear superficie
        surface = pygame.surfarray.make_surface(frame)
        
        return surface
    
    def draw_detections_on_frame(
        self,
        surface: pygame.Surface,
        detections: List,
        scale: float = 1.0
    ):
        """
        Dibuja detecciones sobre el frame de video.
        
        Args:
            surface: Superficie del video
            detections: Lista de objetos Detection
            scale: Factor de escala
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = int(x1 * scale), int(y1 * scale)
            x2, y2 = int(x2 * scale), int(y2 * scale)
            
            # Dibujar rectángulo
            pygame.draw.rect(
                surface,
                (0, 255, 0),
                (x1, y1, x2 - x1, y2 - y1),
                2
            )
            
            # Etiqueta
            if self.font:
                label = f"{det.class_name}: {det.confidence:.0%}"
                text = self.font.render(label, True, (0, 255, 0))
                surface.blit(text, (x1, y1 - 15))
