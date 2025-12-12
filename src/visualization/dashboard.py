"""
Dashboard principal que integra todos los componentes.

Este módulo crea la ventana principal de la aplicación
y coordina la visualización del video, ecosistema y estadísticas.
"""

import pygame
import numpy as np
from typing import Optional, Tuple
import cv2

from .renderer import EcosystemRenderer, VideoRenderer, RenderConfig


class Dashboard:
    """
    Dashboard principal de EcoVision AI.
    
    Integra:
    - Panel de video con detecciones
    - Visualización del ecosistema 2D
    - Indicadores de alerta
    - Estadísticas de evolución
    
    Layout:
    ┌────────────────────────────────────────┐
    │  Video Feed      │   Ecosistema 2D     │
    │  (detecciones)   │   (agentes)         │
    ├──────────────────┴─────────────────────┤
    │  Stats │ Alert │ Fitness Graph         │
    └────────────────────────────────────────┘
    
    Attributes:
        width: Ancho de la ventana
        height: Alto de la ventana
        screen: Ventana principal de Pygame
        clock: Reloj para controlar FPS
    """
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 700,
        title: str = "EcoVision AI - Sistema de Vigilancia Inteligente"
    ):
        """
        Inicializa el dashboard.
        
        Args:
            width: Ancho de la ventana
            height: Alto de la ventana
            title: Título de la ventana
        """
        self.width = width
        self.height = height
        self.title = title
        
        # Componentes (se inicializan en start())
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # Renderizadores
        self.eco_renderer: Optional[EcosystemRenderer] = None
        self.video_renderer: Optional[VideoRenderer] = None
        
        # Superficies
        self.video_surface: Optional[pygame.Surface] = None
        self.eco_surface: Optional[pygame.Surface] = None
        
        # Fuentes
        self.font: Optional[pygame.font.Font] = None
        self.title_font: Optional[pygame.font.Font] = None
        
        # Layout
        self.video_rect = pygame.Rect(10, 40, 580, 400)
        self.eco_rect = pygame.Rect(600, 40, 590, 400)
        self.stats_rect = pygame.Rect(10, 450, 1180, 240)
        
        # Estado
        self.running = False
        self.fps = 30
    
    def start(self):
        """Inicia el dashboard y crea la ventana."""
        pygame.init()
        pygame.font.init()
        
        # Crear ventana
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        
        self.clock = pygame.time.Clock()
        
        # Fuentes
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 18, bold=True)
        
        # Crear superficies para cada panel
        self.video_surface = pygame.Surface(
            (self.video_rect.width, self.video_rect.height)
        )
        self.eco_surface = pygame.Surface(
            (self.eco_rect.width, self.eco_rect.height)
        )
        
        # Inicializar renderizadores
        eco_config = RenderConfig(
            width=self.eco_rect.width,
            height=self.eco_rect.height
        )
        self.eco_renderer = EcosystemRenderer(eco_config)
        self.eco_renderer.initialize(self.eco_surface)
        
        self.video_renderer = VideoRenderer(
            target_size=(self.video_rect.width, self.video_rect.height)
        )
        self.video_renderer.initialize()
        
        self.running = True
        print(f"🖥️ Dashboard iniciado: {self.width}x{self.height}")
    
    def stop(self):
        """Detiene el dashboard y cierra la ventana."""
        self.running = False
        pygame.quit()
        print("🖥️ Dashboard cerrado")
    
    def handle_events(self) -> dict:
        """
        Procesa eventos de Pygame.
        
        Returns:
            Diccionario con eventos procesados
        """
        events = {
            "quit": False,
            "pause": False,
            "reset": False,
            "key": None
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            
            elif event.type == pygame.KEYDOWN:
                events["key"] = event.key
                
                if event.key == pygame.K_ESCAPE:
                    events["quit"] = True
                elif event.key == pygame.K_SPACE:
                    events["pause"] = True
                elif event.key == pygame.K_r:
                    events["reset"] = True
        
        return events
    
    def update(
        self,
        video_frame: Optional[np.ndarray] = None,
        detections: list = None,
        agents: list = None,
        alert_level: float = 0.0,
        alert_category: str = "normal",
        stats: dict = None
    ):
        """
        Actualiza el dashboard con nuevos datos.
        
        Args:
            video_frame: Frame de video (numpy array BGR)
            detections: Lista de objetos Detection
            agents: Lista de objetos Agent
            alert_level: Nivel de alerta (0.0 - 1.0)
            alert_category: Categoría de alerta
            stats: Estadísticas de la simulación
        """
        if not self.screen:
            return
        
        # Limpiar pantalla
        self.screen.fill((25, 25, 35))
        
        # Dibujar título
        self._draw_header()
        
        # Panel de video
        self._draw_video_panel(video_frame, detections or [])
        
        # Panel del ecosistema
        self._draw_ecosystem_panel(
            agents or [],
            detections or [],
            alert_level,
            alert_category
        )
        
        # Panel de estadísticas
        self._draw_stats_panel(stats or {}, alert_level, alert_category)
        
        # Actualizar pantalla
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_header(self):
        """Dibuja el encabezado."""
        # Título
        title = self.title_font.render(
            "🎯 EcoVision AI - Sistema de Vigilancia Inteligente",
            True,
            (220, 220, 220)
        )
        self.screen.blit(title, (10, 10))
        
        # FPS
        fps_text = self.font.render(
            f"FPS: {int(self.clock.get_fps())}",
            True,
            (150, 150, 150)
        )
        self.screen.blit(fps_text, (self.width - 70, 10))
    
    def _draw_video_panel(
        self,
        frame: Optional[np.ndarray],
        detections: list
    ):
        """Dibuja el panel de video."""
        # Borde del panel
        pygame.draw.rect(
            self.screen,
            (60, 60, 70),
            self.video_rect,
            2,
            border_radius=5
        )
        
        # Etiqueta
        label = self.font.render("📹 Video Feed", True, (180, 180, 180))
        self.screen.blit(label, (self.video_rect.x + 5, self.video_rect.y - 18))
        
        if frame is not None:
            # Convertir frame a superficie
            try:
                video_surf = self.video_renderer.frame_to_surface(frame)
                
                # Dibujar detecciones
                scale_x = self.video_rect.width / frame.shape[1]
                scale_y = self.video_rect.height / frame.shape[0]
                
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                    x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                    
                    pygame.draw.rect(
                        video_surf,
                        (0, 255, 0),
                        (x1, y1, x2 - x1, y2 - y1),
                        2
                    )
                
                self.screen.blit(video_surf, self.video_rect.topleft)
                
                # Contador de detecciones
                count_text = self.font.render(
                    f"Detectados: {len(detections)}",
                    True,
                    (0, 255, 0)
                )
                self.screen.blit(
                    count_text,
                    (self.video_rect.x + 5, self.video_rect.bottom - 20)
                )
            except Exception as e:
                # Si hay error, mostrar placeholder
                self._draw_placeholder(self.video_rect, "Video no disponible")
        else:
            self._draw_placeholder(self.video_rect, "Sin video")
    
    def _draw_ecosystem_panel(
        self,
        agents: list,
        detections: list,
        alert_level: float,
        alert_category: str
    ):
        """Dibuja el panel del ecosistema."""
        # Borde del panel
        pygame.draw.rect(
            self.screen,
            (60, 60, 70),
            self.eco_rect,
            2,
            border_radius=5
        )
        
        # Etiqueta
        label = self.font.render("🤖 Ecosistema Evolutivo", True, (180, 180, 180))
        self.screen.blit(label, (self.eco_rect.x + 5, self.eco_rect.y - 18))
        
        # Renderizar ecosistema
        self.eco_renderer.clear()
        
        # Mapear detecciones al espacio del ecosistema
        if detections:
            det_positions = []
            for det in detections:
                # Escalar posiciones al panel del ecosistema
                scale_x = self.eco_rect.width / 640  # Asumiendo video 640px
                scale_y = self.eco_rect.height / 480
                x = det.center[0] * scale_x
                y = det.center[1] * scale_y
                det_positions.append((x, y))
            
            self.eco_renderer.draw_detections(det_positions)
        
        # Dibujar agentes
        if agents:
            # Escalar posiciones de agentes al panel
            for agent in agents:
                # Ajustar escala si el mundo del agente es diferente al panel
                scale_x = self.eco_rect.width / agent.world_size[0]
                scale_y = self.eco_rect.height / agent.world_size[1]
                agent.position[0] *= scale_x
                agent.position[1] *= scale_y
            
            self.eco_renderer.draw_agents(agents, show_trails=True)
            
            # Restaurar posiciones originales
            for agent in agents:
                scale_x = self.eco_rect.width / agent.world_size[0]
                scale_y = self.eco_rect.height / agent.world_size[1]
                agent.position[0] /= scale_x
                agent.position[1] /= scale_y
        
        # Dibujar indicador de alerta en el panel
        self.eco_renderer.draw_alert_indicator(
            alert_level,
            alert_category,
            position=(self.eco_rect.width - 170, 10)
        )
        
        # Copiar superficie al screen
        self.screen.blit(self.eco_surface, self.eco_rect.topleft)
    
    def _draw_stats_panel(
        self,
        stats: dict,
        alert_level: float,
        alert_category: str
    ):
        """Dibuja el panel de estadísticas."""
        x = self.stats_rect.x
        y = self.stats_rect.y
        
        # Fondo
        pygame.draw.rect(
            self.screen,
            (35, 35, 45),
            self.stats_rect,
            border_radius=5
        )
        
        # Sección 1: Estadísticas generales
        self._draw_stat_box(
            "📊 Simulación",
            [
                f"Generación: {stats.get('generation', 0)}",
                f"Paso: {stats.get('step', 0)}/{stats.get('steps_per_gen', 0)}",
                f"Agentes: {stats.get('agent_count', 0)}",
                f"Estado: {'⏸️ Pausado' if stats.get('paused') else '▶️ Activo'}"
            ],
            (x + 10, y + 10)
        )
        
        # Sección 2: Fitness
        self._draw_stat_box(
            "🧬 Evolución",
            [
                f"Mejor Fitness: {stats.get('best_fitness', 0):.1f}",
                f"Promedio: {stats.get('avg_fitness', 0):.1f}",
                f"Mejora: {stats.get('improvement', 0):.1f}",
                f"Detecciones: {stats.get('detections_count', 0)}"
            ],
            (x + 200, y + 10)
        )
        
        # Sección 3: Alerta
        alert_color = EcosystemRenderer.ALERT_COLORS.get(
            alert_category, (128, 128, 128)
        )
        self._draw_alert_gauge(
            alert_level,
            alert_category,
            alert_color,
            (x + 390, y + 10)
        )
        
        # Sección 4: Gráfico de fitness
        best_history = stats.get('best_history', [])
        avg_history = stats.get('avg_history', [])
        self._draw_mini_graph(
            best_history,
            avg_history,
            (x + 580, y + 10),
            (580, 90)
        )
        
        # Controles
        controls = "Controles: [SPACE] Pausar  [R] Reiniciar  [ESC] Salir"
        ctrl_text = self.font.render(controls, True, (120, 120, 120))
        self.screen.blit(ctrl_text, (x + 10, y + 110))
    
    def _draw_stat_box(
        self,
        title: str,
        lines: list,
        position: Tuple[int, int]
    ):
        """Dibuja un cuadro de estadísticas."""
        x, y = position
        
        # Título
        title_surf = self.font.render(title, True, (200, 200, 200))
        self.screen.blit(title_surf, (x, y))
        
        # Líneas
        for i, line in enumerate(lines):
            text = self.font.render(line, True, (160, 160, 160))
            self.screen.blit(text, (x, y + 20 + i * 18))
    
    def _draw_alert_gauge(
        self,
        level: float,
        category: str,
        color: Tuple[int, int, int],
        position: Tuple[int, int]
    ):
        """Dibuja un medidor de alerta visual."""
        x, y = position
        
        # Título
        title = self.font.render("🚨 Nivel de Alerta", True, (200, 200, 200))
        self.screen.blit(title, (x, y))
        
        # Barra grande
        bar_width = 150
        bar_height = 25
        
        pygame.draw.rect(
            self.screen,
            (50, 50, 60),
            (x, y + 22, bar_width, bar_height),
            border_radius=5
        )
        
        fill_width = int(bar_width * level)
        if fill_width > 0:
            pygame.draw.rect(
                self.screen,
                color,
                (x, y + 22, fill_width, bar_height),
                border_radius=5
            )
        
        # Texto
        level_text = self.font.render(
            f"{category.upper()} ({level:.0%})",
            True,
            color
        )
        self.screen.blit(level_text, (x, y + 52))
        
        # Indicador circular
        pygame.draw.circle(
            self.screen,
            color,
            (x + bar_width + 30, y + 35),
            20
        )
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (x + bar_width + 30, y + 35),
            20,
            2
        )
    
    def _draw_mini_graph(
        self,
        best_history: list,
        avg_history: list,
        position: Tuple[int, int],
        size: Tuple[int, int]
    ):
        """Dibuja un mini gráfico de evolución."""
        x, y = position
        width, height = size
        
        # Fondo
        pygame.draw.rect(
            self.screen,
            (45, 45, 55),
            (x, y, width, height),
            border_radius=5
        )
        
        # Título
        title = self.font.render("📈 Evolución del Fitness", True, (200, 200, 200))
        self.screen.blit(title, (x + 5, y + 2))
        
        # Área del gráfico
        graph_x = x + 5
        graph_y = y + 22
        graph_w = width - 10
        graph_h = height - 27
        
        if len(best_history) > 1:
            max_val = max(max(best_history), 1)
            if avg_history:
                max_val = max(max_val, max(avg_history))
            
            # Línea del mejor
            points = []
            for i, val in enumerate(best_history[-100:]):
                px = graph_x + int((i / max(len(best_history[-100:]) - 1, 1)) * graph_w)
                py = graph_y + graph_h - int((val / max_val) * graph_h)
                points.append((px, max(graph_y, min(py, graph_y + graph_h))))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 100), False, points, 2)
            
            # Leyenda
            legend = self.font.render(
                f"Mejor: {best_history[-1]:.1f}" if best_history else "",
                True,
                (0, 255, 100)
            )
            self.screen.blit(legend, (x + width - 100, y + 2))
    
    def _draw_placeholder(self, rect: pygame.Rect, text: str):
        """Dibuja un placeholder con texto."""
        pygame.draw.rect(
            self.screen,
            (40, 40, 50),
            rect,
            border_radius=5
        )
        
        if self.font:
            text_surf = self.font.render(text, True, (100, 100, 100))
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
