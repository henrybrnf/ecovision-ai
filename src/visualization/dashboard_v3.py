"""
Dashboard v3.0 - Interfaz Clara y Profesional

Versión completamente reescrita con:
- Fuentes grandes y claras
- Indicador de alerta prominente con semáforo real
- Colores consistentes por nivel de alerta
- Detecciones coloreadas según alerta
- Layout limpio sin sobreposiciones
"""

import pygame
import numpy as np
import cv2
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Configuración del dashboard."""
    width: int = 1400
    height: int = 850
    fps: int = 30
    title: str = "EcoVision AI v3.0"


class DashboardV3:
    """
    Dashboard con interfaz clara y profesional.
    
    Layout:
    ┌────────────────────────────────────────────────────────┐
    │                     HEADER                              │
    ├───────────────────────┬────────────────────────────────┤
    │                       │                                 │
    │   VIDEO + DETECCIONES │    ECOSISTEMA EVOLUTIVO        │
    │                       │                                 │
    ├───────────────────────┼────────────────────────────────┤
    │   ALERTA (SEMÁFORO)  │    CONTROLES Y ESTADÍSTICAS    │
    └───────────────────────┴────────────────────────────────┘
    """
    
    # Colores claros
    BG_DARK = (20, 22, 30)
    BG_PANEL = (30, 32, 45)
    BG_CARD = (40, 45, 60)
    
    TEXT_WHITE = (255, 255, 255)
    TEXT_LIGHT = (200, 200, 210)
    TEXT_DIM = (120, 125, 140)
    
    ACCENT = (100, 130, 255)
    
    # Colores de alerta (semáforo estándar: 3 colores)
    # ROJO = Emergencia (>70% alerta)
    # AMARILLO = Precaución (30-70% alerta)  
    # VERDE = Normal (<30% alerta)
    ALERT_COLORS = {
        'normal': (0, 200, 80),         # Verde
        'precaución': (255, 200, 0),    # Amarillo
        'emergencia': (255, 50, 50),    # Rojo
    }
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
        # Estado
        self.running = False
        self.paused = False
        self.current_speed = 1
        
        # Pygame
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # Datos
        self.frame = None
        self.detections = []
        self.agents = []
        self.alert_level = 0.0
        self.alert_category = "normal"
        self.stats = {}
        
        # Historial para gráficos
        self.fitness_history = []
        self.detection_history = []
        
        # Fuentes
        self.fonts = {}
        
        # Callbacks
        self.on_pause = None
        self.on_reset = None
        self.on_speed_change = None
        
        # Botones
        self.buttons = []
    
    def start(self):
        """Inicia el dashboard."""
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode(
            (self.config.width, self.config.height),
            pygame.RESIZABLE
        )
        pygame.display.set_caption(self.config.title)
        
        self.clock = pygame.time.Clock()
        
        # Cargar fuentes grandes y claras
        self._load_fonts()
        
        # Crear botones
        self._create_buttons()
        
        self.running = True
        print(f"🖥️ Dashboard v3.0 iniciado: {self.config.width}x{self.config.height}")
    
    def _load_fonts(self):
        """Carga fuentes claras y grandes."""
        try:
            self.fonts['huge'] = pygame.font.SysFont('Arial', 48, bold=True)
            self.fonts['big'] = pygame.font.SysFont('Arial', 28, bold=True)
            self.fonts['title'] = pygame.font.SysFont('Arial', 22, bold=True)
            self.fonts['normal'] = pygame.font.SysFont('Arial', 16)
            self.fonts['small'] = pygame.font.SysFont('Arial', 14)
        except:
            self.fonts['huge'] = pygame.font.Font(None, 52)
            self.fonts['big'] = pygame.font.Font(None, 32)
            self.fonts['title'] = pygame.font.Font(None, 24)
            self.fonts['normal'] = pygame.font.Font(None, 18)
            self.fonts['small'] = pygame.font.Font(None, 16)
    
    def _create_buttons(self):
        """Crea los botones."""
        # Las posiciones se calculan en el render
        self.buttons = []
    
    def stop(self):
        """Detiene el dashboard."""
        self.running = False
        pygame.quit()
        print("🖥️ Dashboard cerrado")
    
    def handle_events(self) -> dict:
        """Procesa eventos."""
        events = {"quit": False}
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events["quit"] = True
                elif event.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_1:
                    self._set_speed(1)
                elif event.key == pygame.K_2:
                    self._set_speed(2)
                elif event.key == pygame.K_3:
                    self._set_speed(3)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
            
            elif event.type == pygame.VIDEORESIZE:
                self.config.width = max(1200, event.w)
                self.config.height = max(700, event.h)
                self.screen = pygame.display.set_mode(
                    (self.config.width, self.config.height),
                    pygame.RESIZABLE
                )
        
        # Procesar clicks en botones
        if mouse_clicked:
            self._handle_button_click(mouse_pos)
        
        return events
    
    def _toggle_pause(self):
        self.paused = not self.paused
        if self.on_pause:
            self.on_pause()
    
    def _reset(self):
        self.fitness_history.clear()
        self.detection_history.clear()
        if self.on_reset:
            self.on_reset()
    
    def _set_speed(self, speed):
        self.current_speed = speed
        if self.on_speed_change:
            self.on_speed_change(speed)
    
    def _handle_button_click(self, pos):
        """Maneja clicks en botones."""
        # Los botones se definen durante el render
        pass
    
    def update(
        self,
        frame=None,
        detections=None,
        agents=None,
        alert_level: float = 0.0,
        alert_category: str = "normal",
        stats: dict = None
    ):
        """Actualiza y renderiza el dashboard."""
        if not self.screen:
            return
        
        self.frame = frame
        self.detections = detections or []
        self.agents = agents or []
        self.alert_level = alert_level
        self.alert_category = alert_category
        self.stats = stats or {}
        
        # Actualizar historial
        self.fitness_history.append(self.stats.get('best_fitness', 0))
        self.detection_history.append(len(self.detections))
        
        # Mantener historial corto
        if len(self.fitness_history) > 100:
            self.fitness_history.pop(0)
            self.detection_history.pop(0)
        
        # Renderizar
        self._render()
        
        pygame.display.flip()
        self.clock.tick(self.config.fps)
    
    def _render(self):
        """Renderiza todo el dashboard."""
        w, h = self.config.width, self.config.height
        
        # Fondo
        self.screen.fill(self.BG_DARK)
        
        # === HEADER ===
        self._draw_header()
        
        # Calcular áreas
        header_h = 60
        margin = 10
        
        # Área superior: Video y Ecosistema
        top_h = (h - header_h - margin * 3) * 0.6
        
        # Panel de Video (izquierda)
        video_rect = pygame.Rect(
            margin, header_h + margin,
            (w - margin * 3) // 2, int(top_h)
        )
        
        # Panel de Ecosistema (derecha)
        eco_rect = pygame.Rect(
            video_rect.right + margin, header_h + margin,
            w - video_rect.right - margin * 2, int(top_h)
        )
        
        # Área inferior: Alerta y Estadísticas
        bottom_y = video_rect.bottom + margin
        bottom_h = h - bottom_y - margin
        
        # Panel de Alerta (izquierda, más grande)
        alert_rect = pygame.Rect(
            margin, bottom_y,
            300, bottom_h
        )
        
        # Panel de Estadísticas (derecha)
        stats_rect = pygame.Rect(
            alert_rect.right + margin, bottom_y,
            w - alert_rect.right - margin * 2, bottom_h
        )
        
        # Dibujar paneles
        self._draw_video_panel(video_rect)
        self._draw_ecosystem_panel(eco_rect)
        self._draw_alert_panel(alert_rect)
        self._draw_stats_panel(stats_rect)
    
    def _draw_header(self):
        """Dibuja el header."""
        header_rect = pygame.Rect(0, 0, self.config.width, 55)
        pygame.draw.rect(self.screen, self.BG_PANEL, header_rect)
        
        # Línea inferior
        pygame.draw.line(
            self.screen, self.ACCENT,
            (0, 54), (self.config.width, 54), 3
        )
        
        # Título
        title = self.fonts['big'].render("🎯 EcoVision AI", True, self.TEXT_WHITE)
        self.screen.blit(title, (20, 12))
        
        # Subtítulo
        subtitle = self.fonts['normal'].render(
            "Sistema de Vigilancia con IA: CNN + Lógica Difusa + Algoritmo Genético",
            True, self.TEXT_LIGHT
        )
        self.screen.blit(subtitle, (220, 18))
        
        # Estado
        status_text = "⏸ PAUSADO" if self.paused else "▶ ACTIVO"
        status_color = (255, 200, 0) if self.paused else (0, 200, 80)
        status = self.fonts['title'].render(status_text, True, status_color)
        self.screen.blit(status, (self.config.width - 150, 15))
        
        # FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        fps = self.fonts['small'].render(fps_text, True, self.TEXT_DIM)
        self.screen.blit(fps, (self.config.width - 70, 38))
    
    def _draw_panel_bg(self, rect, title):
        """Dibuja el fondo de un panel con título."""
        # Fondo
        pygame.draw.rect(self.screen, self.BG_PANEL, rect, border_radius=10)
        
        # Borde
        pygame.draw.rect(self.screen, self.BG_CARD, rect, width=2, border_radius=10)
        
        # Header del panel
        header = pygame.Rect(rect.x, rect.y, rect.width, 35)
        pygame.draw.rect(
            self.screen, self.BG_CARD, header,
            border_top_left_radius=10, border_top_right_radius=10
        )
        
        # Título
        title_surf = self.fonts['title'].render(title, True, self.ACCENT)
        self.screen.blit(title_surf, (rect.x + 15, rect.y + 7))
        
        # Retornar área de contenido
        return pygame.Rect(rect.x + 5, rect.y + 40, rect.width - 10, rect.height - 45)
    
    def _draw_video_panel(self, rect):
        """Dibuja el panel de video con detecciones."""
        content = self._draw_panel_bg(rect, "📹 DETECCIÓN (YOLOv8)")
        
        # Color de alerta actual
        alert_color = self.ALERT_COLORS.get(self.alert_category, (100, 100, 100))
        
        if self.frame is not None:
            # Redimensionar frame
            frame = cv2.resize(self.frame, (content.width, content.height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Dimensiones originales para escalar
            orig_h, orig_w = self.frame.shape[:2]
            h, w = frame.shape[:2]
            
            # Dibujar detecciones con color de alerta
            for det in self.detections:
                x1 = int(det.bbox[0] * w / orig_w)
                y1 = int(det.bbox[1] * h / orig_h)
                x2 = int(det.bbox[2] * w / orig_w)
                y2 = int(det.bbox[3] * h / orig_h)
                
                # Usar color de alerta para las detecciones
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), alert_color, 3)
                
                # Etiqueta
                label = f"PERSONA {det.confidence:.0%}"
                cv2.putText(
                    frame_rgb, label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 2
                )
            
            # Convertir a superficie
            surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(surf, content.topleft)
        else:
            pygame.draw.rect(self.screen, self.BG_CARD, content, border_radius=5)
            no_video = self.fonts['normal'].render(
                "Sin video - Esperando fuente...",
                True, self.TEXT_DIM
            )
            self.screen.blit(no_video, (content.centerx - 100, content.centery))
        
        # Contador de detecciones
        count_text = f"Detectadas: {len(self.detections)} personas"
        count = self.fonts['normal'].render(count_text, True, alert_color)
        self.screen.blit(count, (content.x + 5, content.bottom + 5))
    
    def _draw_ecosystem_panel(self, rect):
        """Dibuja el panel del ecosistema."""
        content = self._draw_panel_bg(rect, "🧬 ECOSISTEMA EVOLUTIVO")
        
        # Fondo del mundo
        pygame.draw.rect(self.screen, (25, 28, 38), content, border_radius=5)
        
        # Grid suave
        for x in range(content.x, content.right, 50):
            pygame.draw.line(self.screen, (35, 40, 55), (x, content.y), (x, content.bottom))
        for y in range(content.y, content.bottom, 50):
            pygame.draw.line(self.screen, (35, 40, 55), (content.x, y), (content.right, y))
        
        # Color de alerta para objetivos
        alert_color = self.ALERT_COLORS.get(self.alert_category, (100, 100, 100))
        
        # Dibujar objetivos (detecciones mapeadas)
        if self.detections:
            for det in self.detections:
                # Mapear posición
                x = content.x + int(det.center[0] * content.width / 768)
                y = content.y + int(det.center[1] * content.height / 432)
                x = max(content.x + 10, min(x, content.right - 10))
                y = max(content.y + 10, min(y, content.bottom - 10))
                
                # Objetivo con color de alerta
                pygame.draw.circle(self.screen, alert_color, (x, y), 20, 3)
                pygame.draw.circle(self.screen, alert_color, (x, y), 10, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 4)
        
        # Dibujar agentes
        for agent in self.agents:
            scale_x = content.width / agent.world_size[0]
            scale_y = content.height / agent.world_size[1]
            
            ax = content.x + int(agent.position[0] * scale_x)
            ay = content.y + int(agent.position[1] * scale_y)
            ax = max(content.x + 8, min(ax, content.right - 8))
            ay = max(content.y + 8, min(ay, content.bottom - 8))
            
            # Campo de visión
            angle = agent.angle
            for da in [-0.5, 0, 0.5]:
                end_x = ax + int(np.cos(angle + da) * 20)
                end_y = ay + int(np.sin(angle + da) * 20)
                pygame.draw.line(self.screen, (60, 65, 85), (ax, ay), (end_x, end_y))
            
            # Cuerpo
            pygame.draw.circle(self.screen, agent.color, (ax, ay), 7)
            pygame.draw.circle(self.screen, self.TEXT_WHITE, (ax, ay), 7, 2)
            
            # Dirección
            dir_x = ax + int(np.cos(angle) * 12)
            dir_y = ay + int(np.sin(angle) * 12)
            pygame.draw.line(self.screen, self.TEXT_WHITE, (ax, ay), (dir_x, dir_y), 2)
        
        # Leyenda
        gen = self.stats.get('generation', 0)
        step = self.stats.get('step', 0)
        legend = f"Generación: {gen} | Paso: {step} | Agentes: {len(self.agents)}"
        leg_surf = self.fonts['small'].render(legend, True, self.TEXT_LIGHT)
        self.screen.blit(leg_surf, (content.x + 5, content.bottom + 5))
    
    def _draw_alert_panel(self, rect):
        """Dibuja el panel de alerta con semáforo de 3 colores."""
        content = self._draw_panel_bg(rect, "🚦 ALERTA")
        
        # Color actual según categoría
        alert_color = self.ALERT_COLORS.get(self.alert_category, (100, 100, 100))
        
        # === SEMÁFORO ESTÁNDAR (3 LUCES) ===
        sem_x = content.x + 50
        sem_y = content.y + 10
        
        # Fondo del semáforo
        sem_bg = pygame.Rect(sem_x - 25, sem_y, 50, 130)
        pygame.draw.rect(self.screen, (25, 25, 30), sem_bg, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 70), sem_bg, width=2, border_radius=8)
        
        # Las 3 luces: ROJO (arriba), AMARILLO (medio), VERDE (abajo)
        luz_rojo_y = sem_y + 25
        luz_amarillo_y = sem_y + 65
        luz_verde_y = sem_y + 105
        
        # Determinar qué luz está encendida
        luz_rojo_on = self.alert_category == 'emergencia'
        luz_amarillo_on = self.alert_category == 'precaución'
        luz_verde_on = self.alert_category == 'normal'
        
        # Dibujar luz ROJA
        rojo = self.ALERT_COLORS['emergencia'] if luz_rojo_on else (60, 20, 20)
        pygame.draw.circle(self.screen, rojo, (sem_x, luz_rojo_y), 18)
        pygame.draw.circle(self.screen, (80, 80, 90), (sem_x, luz_rojo_y), 18, 2)
        
        # Dibujar luz AMARILLA
        amarillo = self.ALERT_COLORS['precaución'] if luz_amarillo_on else (60, 50, 10)
        pygame.draw.circle(self.screen, amarillo, (sem_x, luz_amarillo_y), 18)
        pygame.draw.circle(self.screen, (80, 80, 90), (sem_x, luz_amarillo_y), 18, 2)
        
        # Dibujar luz VERDE
        verde = self.ALERT_COLORS['normal'] if luz_verde_on else (10, 50, 20)
        pygame.draw.circle(self.screen, verde, (sem_x, luz_verde_y), 18)
        pygame.draw.circle(self.screen, (80, 80, 90), (sem_x, luz_verde_y), 18, 2)
        
        # === INFORMACIÓN A LA DERECHA DEL SEMÁFORO ===
        info_x = sem_x + 50
        info_y = sem_y + 5
        
        # Porcentaje grande
        pct_text = f"{self.alert_level:.0%}"
        pct_surf = self.fonts['big'].render(pct_text, True, alert_color)
        self.screen.blit(pct_surf, (info_x, info_y))
        
        # Categoría
        cat_names = {'normal': 'NORMAL', 'precaución': 'PRECAUCIÓN', 'emergencia': 'EMERGENCIA'}
        cat_text = cat_names.get(self.alert_category, 'NORMAL')
        cat_surf = self.fonts['normal'].render(cat_text, True, alert_color)
        self.screen.blit(cat_surf, (info_x, info_y + 35))
        
        # Descripción
        descriptions = {
            'normal': 'Situación tranquila',
            'precaución': 'Vigilar situación',
            'emergencia': '¡Acción requerida!'
        }
        desc = descriptions.get(self.alert_category, '')
        desc_surf = self.fonts['small'].render(desc, True, self.TEXT_LIGHT)
        self.screen.blit(desc_surf, (info_x, info_y + 55))
        
        # === LEYENDA DE COLORES ===
        ley_y = content.y + 150
        
        leyenda_title = self.fonts['small'].render("Significado:", True, self.TEXT_DIM)
        self.screen.blit(leyenda_title, (content.x + 10, ley_y))
        ley_y += 18
        
        leyenda = [
            ("🔴", "ROJO", ">70%", "Emergencia"),
            ("🟡", "AMAR", "30-70%", "Precaución"),
            ("🟢", "VERDE", "<30%", "Normal"),
        ]
        
        for emoji, color_name, rango, estado in leyenda:
            # Emoji como indicador
            line = f"{emoji} {rango} = {estado}"
            line_surf = self.fonts['small'].render(line, True, self.TEXT_LIGHT)
            self.screen.blit(line_surf, (content.x + 10, ley_y))
            ley_y += 16
        
        # === CÓMO SE CALCULA ===
        calc_y = ley_y + 10
        
        calc_title = self.fonts['small'].render("Cálculo:", True, self.TEXT_DIM)
        self.screen.blit(calc_title, (content.x + 10, calc_y))
        calc_y += 18
        
        calc_lines = [
            f"Personas: {len(self.detections)}",
            f"Densidad: {len(self.detections)/12:.0%}",
        ]
        
        for line in calc_lines:
            line_surf = self.fonts['small'].render(line, True, self.TEXT_LIGHT)
            self.screen.blit(line_surf, (content.x + 10, calc_y))
            calc_y += 15
    
    def _draw_stats_panel(self, rect):
        """Dibuja el panel de estadísticas y controles."""
        content = self._draw_panel_bg(rect, "📊 ESTADÍSTICAS Y CONTROLES")
        
        # Dividir en columnas
        col_w = content.width // 3
        
        # === COLUMNA 1: ESTADÍSTICAS ===
        col1_x = content.x + 10
        y = content.y + 10
        
        stats_title = self.fonts['normal'].render("ESTADÍSTICAS", True, self.ACCENT)
        self.screen.blit(stats_title, (col1_x, y))
        y += 25
        
        stats_data = [
            ("Frames:", self.stats.get('frame_count', 0)),
            ("Personas:", len(self.detections)),
            ("Generación:", self.stats.get('generation', 0)),
            ("Mejor Fitness:", f"{self.stats.get('best_fitness', 0):.1f}"),
            ("Fitness Prom:", f"{self.stats.get('avg_fitness', 0):.1f}"),
        ]
        
        for label, value in stats_data:
            label_surf = self.fonts['small'].render(label, True, self.TEXT_DIM)
            value_surf = self.fonts['small'].render(str(value), True, self.TEXT_WHITE)
            self.screen.blit(label_surf, (col1_x, y))
            self.screen.blit(value_surf, (col1_x + 100, y))
            y += 20
        
        # === COLUMNA 2: CONTROLES ===
        col2_x = content.x + col_w + 10
        y = content.y + 10
        
        ctrl_title = self.fonts['normal'].render("CONTROLES", True, self.ACCENT)
        self.screen.blit(ctrl_title, (col2_x, y))
        y += 25
        
        # Velocidad actual
        speed_text = f"Velocidad: x{self.current_speed}"
        speed_surf = self.fonts['normal'].render(speed_text, True, self.TEXT_WHITE)
        self.screen.blit(speed_surf, (col2_x, y))
        y += 25
        
        # Botones de velocidad
        btn_colors = [(80, 100, 180), (100, 80, 180), (180, 80, 100)]
        for i, speed in enumerate([1, 2, 3]):
            btn_rect = pygame.Rect(col2_x + i * 55, y, 50, 30)
            color = btn_colors[i] if self.current_speed == speed else self.BG_CARD
            
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.TEXT_DIM, btn_rect, width=1, border_radius=5)
            
            btn_text = self.fonts['small'].render(f"x{speed}", True, self.TEXT_WHITE)
            btn_text_rect = btn_text.get_rect(center=btn_rect.center)
            self.screen.blit(btn_text, btn_text_rect)
        
        y += 45
        
        # Atajos
        shortcuts = [
            ("[SPACE]", "Pausar"),
            ("[R]", "Reiniciar"),
            ("[1/2/3]", "Velocidad"),
            ("[ESC]", "Salir"),
        ]
        
        for key, desc in shortcuts:
            key_surf = self.fonts['small'].render(key, True, self.ACCENT)
            desc_surf = self.fonts['small'].render(desc, True, self.TEXT_DIM)
            self.screen.blit(key_surf, (col2_x, y))
            self.screen.blit(desc_surf, (col2_x + 70, y))
            y += 18
        
        # === COLUMNA 3: GRÁFICOS ===
        col3_x = content.x + col_w * 2 + 10
        graph_w = col_w - 20
        graph_h = (content.height - 60) // 2
        
        # Gráfico de Fitness
        self._draw_mini_graph(
            pygame.Rect(col3_x, content.y + 10, graph_w, graph_h - 5),
            self.fitness_history,
            "EVOLUCIÓN FITNESS",
            (0, 200, 80)
        )
        
        # Gráfico de Detecciones
        self._draw_mini_graph(
            pygame.Rect(col3_x, content.y + graph_h + 15, graph_w, graph_h - 5),
            self.detection_history,
            "PERSONAS DETECTADAS",
            (100, 130, 255)
        )
    
    def _draw_mini_graph(self, rect, data, title, color):
        """Dibuja un mini gráfico."""
        # Fondo
        pygame.draw.rect(self.screen, self.BG_CARD, rect, border_radius=5)
        
        # Título
        title_surf = self.fonts['small'].render(title, True, color)
        self.screen.blit(title_surf, (rect.x + 5, rect.y + 3))
        
        # Área del gráfico
        graph_rect = pygame.Rect(rect.x + 5, rect.y + 22, rect.width - 10, rect.height - 27)
        
        # Grid
        for i in range(4):
            gy = graph_rect.y + int(i * graph_rect.height / 3)
            pygame.draw.line(
                self.screen, (50, 55, 70),
                (graph_rect.x, gy), (graph_rect.right, gy)
            )
        
        # Datos
        if len(data) > 1:
            max_val = max(data) if max(data) > 0 else 1
            
            points = []
            for i, val in enumerate(data[-50:]):
                x = graph_rect.x + int(i * graph_rect.width / max(len(data[-50:]) - 1, 1))
                y = graph_rect.bottom - int((val / max_val) * graph_rect.height)
                y = max(graph_rect.y, min(y, graph_rect.bottom))
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            
            # Valor actual
            if data:
                val_text = f"{data[-1]:.1f}" if isinstance(data[-1], float) else str(data[-1])
                val_surf = self.fonts['small'].render(val_text, True, color)
                self.screen.blit(val_surf, (rect.right - 40, rect.y + 3))
