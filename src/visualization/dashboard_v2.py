"""
Dashboard Principal Mejorado v2.0

Dashboard con interfaz moderna, botones interactivos,
gráficos en tiempo real y mejor visualización.
"""

import pygame
import numpy as np
import cv2
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .ui_components import (
    Colors, Button, Panel, Chart, AlertIndicator, 
    StatCard, ProgressBar, ButtonStyle
)


@dataclass
class DashboardConfig:
    """Configuración del dashboard."""
    width: int = 1400
    height: int = 800
    fps: int = 30
    title: str = "EcoVision AI v2.0"
    resizable: bool = True


class DashboardV2:
    """
    Dashboard mejorado con interfaz moderna.
    
    Features:
    - Ventana redimensionable
    - Botones interactivos con hover
    - Paneles organizados
    - Gráficos en tiempo real
    - Estadísticas visuales
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
        # Estado
        self.running = False
        self.paused = False
        
        # Pygame
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # UI Components
        self.panels: List[Panel] = []
        self.buttons: List[Button] = []
        self.charts: List[Chart] = []
        self.stats: List[StatCard] = []
        self.alert_indicator: Optional[AlertIndicator] = None
        self.gen_progress: Optional[ProgressBar] = None
        
        # Data
        self.current_frame: Optional[np.ndarray] = None
        self.detections: List = []
        self.agents: List = []
        self.alert_level = 0.0
        self.alert_category = "normal"
        self.sim_stats = {}
        
        # Fonts
        self.fonts = {}
        
        # Event callbacks
        self.on_pause = None
        self.on_reset = None
        self.on_speed_change = None
    
    def start(self):
        """Inicia el dashboard."""
        pygame.init()
        pygame.font.init()
        
        # Flags para ventana redimensionable
        flags = pygame.RESIZABLE if self.config.resizable else 0
        
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height),
            flags
        )
        pygame.display.set_caption(self.config.title)
        
        self.clock = pygame.time.Clock()
        
        # Cargar fuentes
        self._load_fonts()
        
        # Crear UI
        self._create_ui()
        
        self.running = True
        print(f"🖥️ Dashboard v2.0 iniciado: {self.config.width}x{self.config.height}")
    
    def _load_fonts(self):
        """Carga las fuentes."""
        try:
            self.fonts['title'] = pygame.font.SysFont('Arial', 26, bold=True)
            self.fonts['subtitle'] = pygame.font.SysFont('Arial', 15)
            self.fonts['normal'] = pygame.font.SysFont('Arial', 14)
            self.fonts['small'] = pygame.font.SysFont('Arial', 12)
            self.fonts['tiny'] = pygame.font.SysFont('Arial', 10)
        except:
            self.fonts['title'] = pygame.font.Font(None, 30)
            self.fonts['subtitle'] = pygame.font.Font(None, 18)
            self.fonts['normal'] = pygame.font.Font(None, 16)
            self.fonts['small'] = pygame.font.Font(None, 14)
            self.fonts['tiny'] = pygame.font.Font(None, 12)
        
        # Variable para velocidad actual
        self.current_speed = 1
    
    def _create_ui(self):
        """Crea los componentes de UI."""
        w, h = self.config.width, self.config.height
        
        # Panel de Video
        self.video_panel = Panel(
            pygame.Rect(10, 60, w//2 - 15, h//2 - 30),
            title="📹 Detección de Personas (YOLO)"
        )
        self.panels.append(self.video_panel)
        
        # Panel del Ecosistema
        self.eco_panel = Panel(
            pygame.Rect(w//2 + 5, 60, w//2 - 15, h//2 - 30),
            title="🤖 Ecosistema Evolutivo"
        )
        self.panels.append(self.eco_panel)
        
        # Panel de Control
        self.control_panel = Panel(
            pygame.Rect(10, h//2 + 40, 200, h//2 - 50),
            title="⚙️ Controles"
        )
        self.panels.append(self.control_panel)
        
        # Panel de Alerta
        self.alert_panel = Panel(
            pygame.Rect(220, h//2 + 40, 180, h//2 - 50),
            title="🚦 Alerta"
        )
        self.panels.append(self.alert_panel)
        
        # Panel de Estadísticas
        self.stats_panel = Panel(
            pygame.Rect(410, h//2 + 40, w - 420, h//2 - 50),
            title="📊 Estadísticas y Evolución"
        )
        self.panels.append(self.stats_panel)
        
        # Botones de control
        ctrl = self.control_panel.content_rect
        
        # Botón Pausar
        self.btn_pause = Button(
            pygame.Rect(ctrl.x + 5, ctrl.y + 10, ctrl.width - 10, 35),
            "⏸️ PAUSAR",
            callback=self._on_pause_click,
            style=ButtonStyle(bg_color=Colors.PRIMARY)
        )
        self.buttons.append(self.btn_pause)
        
        # Botón Reiniciar
        self.btn_reset = Button(
            pygame.Rect(ctrl.x + 5, ctrl.y + 55, ctrl.width - 10, 35),
            "🔄 REINICIAR",
            callback=self._on_reset_click,
            style=ButtonStyle(bg_color=Colors.WARNING)
        )
        self.buttons.append(self.btn_reset)
        
        # Botón Velocidad x1
        self.btn_speed1 = Button(
            pygame.Rect(ctrl.x + 5, ctrl.y + 100, 55, 30),
            "x1",
            callback=lambda: self._on_speed_click(1),
            style=ButtonStyle(bg_color=Colors.INFO, font_size=12)
        )
        self.buttons.append(self.btn_speed1)
        
        # Botón Velocidad x2
        self.btn_speed2 = Button(
            pygame.Rect(ctrl.x + 65, ctrl.y + 100, 55, 30),
            "x2",
            callback=lambda: self._on_speed_click(2),
            style=ButtonStyle(bg_color=Colors.INFO, font_size=12)
        )
        self.buttons.append(self.btn_speed2)
        
        # Botón Velocidad x3
        self.btn_speed3 = Button(
            pygame.Rect(ctrl.x + 125, ctrl.y + 100, 55, 30),
            "x3",
            callback=lambda: self._on_speed_click(3),
            style=ButtonStyle(bg_color=Colors.INFO, font_size=12)
        )
        self.buttons.append(self.btn_speed3)
        
        # Indicador de Alerta
        alert_rect = self.alert_panel.content_rect
        self.alert_indicator = AlertIndicator(
            pygame.Rect(alert_rect.x, alert_rect.y, alert_rect.width, 130)
        )
        
        # Barra de progreso de generación
        self.gen_progress = ProgressBar(
            pygame.Rect(alert_rect.x + 5, alert_rect.bottom - 30, alert_rect.width - 10, 20),
            max_value=100,
            color=Colors.SECONDARY
        )
        
        # Stats Cards
        stats = self.stats_panel.content_rect
        card_w = (stats.width - 40) // 4
        
        self.stat_frames = StatCard(
            pygame.Rect(stats.x + 5, stats.y + 5, card_w, 70),
            "Frames", icon="🖼️", color=Colors.PRIMARY
        )
        self.stats.append(self.stat_frames)
        
        self.stat_persons = StatCard(
            pygame.Rect(stats.x + card_w + 15, stats.y + 5, card_w, 70),
            "Personas", icon="👥", color=Colors.SUCCESS
        )
        self.stats.append(self.stat_persons)
        
        self.stat_gen = StatCard(
            pygame.Rect(stats.x + 2*card_w + 25, stats.y + 5, card_w, 70),
            "Generación", icon="🧬", color=Colors.SECONDARY
        )
        self.stats.append(self.stat_gen)
        
        self.stat_fitness = StatCard(
            pygame.Rect(stats.x + 3*card_w + 35, stats.y + 5, card_w, 70),
            "Mejor Fitness", icon="💪", color=Colors.ACCENT
        )
        self.stats.append(self.stat_fitness)
        
        # Gráfico de Fitness
        self.fitness_chart = Chart(
            pygame.Rect(stats.x + 5, stats.y + 85, (stats.width - 20)//2, stats.height - 95),
            max_points=100,
            color=Colors.SUCCESS,
            title="Evolución del Fitness"
        )
        self.charts.append(self.fitness_chart)
        
        # Gráfico de Detecciones
        self.detection_chart = Chart(
            pygame.Rect(stats.x + (stats.width - 20)//2 + 15, stats.y + 85, 
                       (stats.width - 20)//2, stats.height - 95),
            max_points=100,
            color=Colors.PRIMARY,
            title="Personas Detectadas"
        )
        self.charts.append(self.detection_chart)
    
    def _on_pause_click(self):
        """Manejador del botón pausar."""
        self.paused = not self.paused
        self.btn_pause.text = "▶️ CONTINUAR" if self.paused else "⏸️ PAUSAR"
        if self.on_pause:
            self.on_pause()
    
    def _on_reset_click(self):
        """Manejador del botón reiniciar."""
        if self.on_reset:
            self.on_reset()
        self.fitness_chart.clear()
        self.detection_chart.clear()
    
    def _on_speed_click(self, speed: int):
        """Manejador de botones de velocidad."""
        self.current_speed = speed
        if self.on_speed_change:
            self.on_speed_change(speed)
    
    def stop(self):
        """Detiene el dashboard."""
        self.running = False
        pygame.quit()
        print("🖥️ Dashboard cerrado")
    
    def handle_events(self) -> dict:
        """Procesa eventos."""
        events = {"quit": False, "pause": False, "reset": False}
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events["quit"] = True
                elif event.key == pygame.K_SPACE:
                    self._on_pause_click()
                elif event.key == pygame.K_r:
                    self._on_reset_click()
            
            # Procesar botones
            for btn in self.buttons:
                btn.handle_event(event)
        
        return events
    
    def _handle_resize(self, new_w: int, new_h: int):
        """Maneja el redimensionamiento de ventana."""
        self.config.width = max(1200, new_w)
        self.config.height = max(700, new_h)
        
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height),
            pygame.RESIZABLE
        )
        
        # Recrear UI con nuevo tamaño
        self.panels.clear()
        self.buttons.clear()
        self.stats.clear()
        self.charts.clear()
        self._create_ui()
    
    def update(
        self,
        frame: Optional[np.ndarray] = None,
        detections: List = None,
        agents: List = None,
        alert_level: float = 0.0,
        alert_category: str = "normal",
        stats: dict = None
    ):
        """Actualiza el dashboard."""
        if not self.screen:
            return
        
        self.current_frame = frame
        self.detections = detections or []
        self.agents = agents or []
        self.alert_level = alert_level
        self.alert_category = alert_category
        self.sim_stats = stats or {}
        
        # Actualizar componentes animados
        self.alert_indicator.set_alert(alert_level, alert_category)
        self.alert_indicator.update()
        
        # Actualizar estadísticas
        self.stat_frames.set_value(stats.get('frame_count', 0))
        self.stat_persons.set_value(len(self.detections))
        self.stat_gen.set_value(stats.get('generation', 0))
        self.stat_fitness.set_value(f"{stats.get('best_fitness', 0):.1f}")
        
        # Actualizar gráficos
        self.fitness_chart.add_point(stats.get('best_fitness', 0))
        self.detection_chart.add_point(len(self.detections))
        
        # Actualizar barra de progreso
        step = stats.get('step', 0)
        steps_per_gen = stats.get('steps_per_gen', 100)
        self.gen_progress.set_value((step / steps_per_gen) * 100)
        self.gen_progress.update()
        
        # Renderizar
        self._render()
        
        pygame.display.flip()
        self.clock.tick(self.config.fps)
    
    def _render(self):
        """Renderiza todos los componentes."""
        # Fondo con gradiente
        self._draw_gradient_background()
        
        # Header
        self._draw_header()
        
        # Paneles
        for panel in self.panels:
            panel.draw(self.screen)
        
        # Contenido de paneles
        self._draw_video_content()
        self._draw_ecosystem_content()
        self._draw_control_info()
        
        # Indicador de alerta
        self.alert_indicator.draw(self.screen)
        
        # Barra de progreso
        self.gen_progress.draw(self.screen)
        
        # Stats
        for stat in self.stats:
            stat.draw(self.screen)
        
        # Gráficos
        for chart in self.charts:
            chart.draw(self.screen)
        
        # Botones (al final para que estén encima)
        for btn in self.buttons:
            btn.draw(self.screen)
    
    def _draw_gradient_background(self):
        """Dibuja un fondo con gradiente."""
        for y in range(self.config.height):
            ratio = y / self.config.height
            r = int(15 + ratio * 10)
            g = int(15 + ratio * 10)
            b = int(25 + ratio * 15)
            pygame.draw.line(
                self.screen, (r, g, b),
                (0, y), (self.config.width, y)
            )
    
    def _draw_header(self):
        """Dibuja el header."""
        # Barra de título
        header_rect = pygame.Rect(0, 0, self.config.width, 50)
        pygame.draw.rect(self.screen, Colors.BG_PANEL, header_rect)
        pygame.draw.line(
            self.screen, Colors.PRIMARY,
            (0, 49), (self.config.width, 49), 2
        )
        
        # Título
        title = self.fonts['title'].render("🎯 EcoVision AI", True, Colors.TEXT_PRIMARY)
        self.screen.blit(title, (15, 8))
        
        # Subtítulo
        subtitle = self.fonts['subtitle'].render(
            "Sistema Inteligente de Vigilancia con IA",
            True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(subtitle, (200, 15))
        
        # FPS
        fps = self.fonts['normal'].render(
            f"FPS: {int(self.clock.get_fps())}",
            True, Colors.TEXT_MUTED
        )
        self.screen.blit(fps, (self.config.width - 80, 15))
        
        # Estado
        status = "⏸️ PAUSADO" if self.paused else "▶️ ACTIVO"
        status_color = Colors.WARNING if self.paused else Colors.SUCCESS
        status_text = self.fonts['normal'].render(status, True, status_color)
        self.screen.blit(status_text, (self.config.width - 180, 15))
    
    def _draw_video_content(self):
        """Dibuja el contenido del panel de video."""
        rect = self.video_panel.content_rect
        
        if self.current_frame is not None:
            # Redimensionar frame
            frame = cv2.resize(self.current_frame, (rect.width, rect.height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Dibujar detecciones
            h, w = frame.shape[:2]
            orig_h, orig_w = self.current_frame.shape[:2]
            
            for det in self.detections:
                x1 = int(det.bbox[0] * w / orig_w)
                y1 = int(det.bbox[1] * h / orig_h)
                x2 = int(det.bbox[2] * w / orig_w)
                y2 = int(det.bbox[3] * h / orig_h)
                
                # Color según confianza
                if det.confidence > 0.5:
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_rgb, f"{det.confidence:.0%}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Convertir a superficie pygame
            surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(surf, rect.topleft)
        else:
            # Placeholder
            pygame.draw.rect(self.screen, Colors.BG_CARD, rect, border_radius=5)
            text = self.fonts['normal'].render("Sin video", True, Colors.TEXT_MUTED)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        # Info de detecciones
        info = self.fonts['small'].render(
            f"Detectadas: {len(self.detections)} personas",
            True, Colors.SUCCESS
        )
        self.screen.blit(info, (rect.x + 5, rect.bottom + 5))
    
    def _draw_ecosystem_content(self):
        """Dibuja el contenido del ecosistema."""
        rect = self.eco_panel.content_rect
        
        # Fondo del mundo
        pygame.draw.rect(self.screen, (25, 25, 35), rect, border_radius=5)
        
        # Grid
        for x in range(rect.x, rect.right, 40):
            pygame.draw.line(self.screen, (40, 40, 55), (x, rect.y), (x, rect.bottom))
        for y in range(rect.y, rect.bottom, 40):
            pygame.draw.line(self.screen, (40, 40, 55), (rect.x, y), (rect.right, y))
        
        # Dibujar detecciones mapeadas como objetivos
        if self.detections:
            for det in self.detections:
                # Mapear posición
                x = rect.x + int(det.center[0] * rect.width / 768)
                y = rect.y + int(det.center[1] * rect.height / 432)
                x = max(rect.x, min(x, rect.right - 10))
                y = max(rect.y, min(y, rect.bottom - 10))
                
                # Círculo de objetivo
                pygame.draw.circle(self.screen, (100, 0, 0), (x, y), 18, 2)
                pygame.draw.circle(self.screen, (200, 50, 50), (x, y), 12, 2)
                pygame.draw.circle(self.screen, (255, 100, 100), (x, y), 6)
        
        # Dibujar agentes
        for agent in self.agents:
            # Escalar posición al rect del panel
            scale_x = rect.width / agent.world_size[0]
            scale_y = rect.height / agent.world_size[1]
            
            ax = rect.x + int(agent.position[0] * scale_x)
            ay = rect.y + int(agent.position[1] * scale_y)
            ax = max(rect.x + 5, min(ax, rect.right - 5))
            ay = max(rect.y + 5, min(ay, rect.bottom - 5))
            
            # Campo de visión (cono)
            angle = agent.angle
            cone_len = 25
            for da in [-0.4, 0, 0.4]:
                end_x = ax + int(np.cos(angle + da) * cone_len)
                end_y = ay + int(np.sin(angle + da) * cone_len)
                pygame.draw.line(self.screen, (60, 60, 80), (ax, ay), (end_x, end_y))
            
            # Cuerpo del agente
            color = agent.color
            pygame.draw.circle(self.screen, color, (ax, ay), 8)
            pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), 8, 1)
            
            # Dirección
            dir_x = ax + int(np.cos(angle) * 12)
            dir_y = ay + int(np.sin(angle) * 12)
            pygame.draw.line(self.screen, (255, 255, 255), (ax, ay), (dir_x, dir_y), 2)
        
        # Leyenda
        legend = self.fonts['small'].render(
            f"Agentes: {len(self.agents)} | Gen: {self.sim_stats.get('generation', 0)}",
            True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(legend, (rect.x + 5, rect.bottom + 5))
    
    def _draw_control_info(self):
        """Dibuja información adicional en el panel de control."""
        rect = self.control_panel.content_rect
        
        # Etiqueta de velocidad sobre los botones
        speed_label = self.fonts['small'].render(
            f"Velocidad: x{self.current_speed}",
            True, Colors.TEXT_PRIMARY
        )
        self.screen.blit(speed_label, (rect.x + 5, rect.y + 138))
        
        # Línea separadora
        y = rect.y + 160
        pygame.draw.line(
            self.screen, Colors.BORDER,
            (rect.x + 5, y), (rect.right - 5, y)
        )
        y += 10
        
        # Título de atajos
        shortcuts_title = self.fonts['small'].render(
            "Atajos de Teclado:",
            True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(shortcuts_title, (rect.x + 5, y))
        y += 20
        
        # Información de teclas
        info_lines = [
            ("SPACE", "Pausar"),
            ("R", "Reiniciar"),
            ("ESC", "Salir"),
        ]
        
        for key, desc in info_lines:
            key_text = self.fonts['small'].render(f"[{key}]", True, Colors.PRIMARY)
            desc_text = self.fonts['small'].render(desc, True, Colors.TEXT_MUTED)
            self.screen.blit(key_text, (rect.x + 5, y))
            self.screen.blit(desc_text, (rect.x + 60, y))
            y += 16
        
        # Línea separadora
        y += 5
        pygame.draw.line(
            self.screen, Colors.BORDER,
            (rect.x + 5, y), (rect.right - 5, y)
        )
        y += 10
        
        # Título de tecnologías
        tech_title = self.fonts['small'].render(
            "Tecnologías IA:",
            True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(tech_title, (rect.x + 5, y))
        y += 18
        
        info = [
            ("CNN", "YOLOv8"),
            ("Fuzzy", "Alertas"),
            ("GA", "Evolución"),
            ("NN", "Cerebros"),
        ]
        
        for tech, desc in info:
            tech_text = self.fonts['tiny'].render(tech, True, Colors.INFO)
            desc_text = self.fonts['tiny'].render(f": {desc}", True, Colors.TEXT_MUTED)
            self.screen.blit(tech_text, (rect.x + 5, y))
            self.screen.blit(desc_text, (rect.x + 45, y))
            y += 14

