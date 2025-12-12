"""
Componentes UI interactivos para Pygame.

Este módulo proporciona widgets reutilizables:
- Button: Botones con estados (normal, hover, pressed)
- Panel: Contenedores con bordes y sombras
- ProgressBar: Barras de progreso animadas
- Chart: Gráficos simples (líneas)
- AlertIndicator: Indicador visual de alerta
- Slider: Control deslizante
"""

import pygame
import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass


# Paleta de colores moderna
class Colors:
    """Paleta de colores del tema."""
    # Fondos
    BG_DARK = (15, 15, 25)
    BG_PANEL = (25, 25, 40)
    BG_CARD = (35, 35, 55)
    
    # Acentos
    PRIMARY = (102, 126, 234)       # Azul violeta
    SECONDARY = (118, 75, 162)      # Púrpura
    ACCENT = (240, 147, 251)        # Rosa
    
    # Estados
    SUCCESS = (40, 167, 69)         # Verde
    WARNING = (255, 193, 7)         # Amarillo
    DANGER = (220, 53, 69)          # Rojo
    INFO = (23, 162, 184)           # Cyan
    
    # Alertas
    ALERT_NORMAL = (40, 200, 100)
    ALERT_PRECAUCION = (255, 200, 0)
    ALERT_ALERTA = (255, 140, 0)
    ALERT_EMERGENCIA = (255, 50, 50)
    
    # Texto
    TEXT_PRIMARY = (240, 240, 250)
    TEXT_SECONDARY = (160, 160, 180)
    TEXT_MUTED = (100, 100, 120)
    
    # Bordes
    BORDER = (60, 60, 80)
    BORDER_LIGHT = (80, 80, 100)


@dataclass
class ButtonStyle:
    """Estilo de un botón."""
    bg_color: Tuple[int, int, int] = Colors.PRIMARY
    hover_color: Tuple[int, int, int] = (130, 150, 255)
    pressed_color: Tuple[int, int, int] = (80, 100, 200)
    text_color: Tuple[int, int, int] = Colors.TEXT_PRIMARY
    border_radius: int = 8
    font_size: int = 14


class Button:
    """
    Botón interactivo con estados visuales.
    
    Soporta:
    - Hover (mouse encima)
    - Pressed (click)
    - Disabled (deshabilitado)
    - Icono opcional
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        callback: Optional[Callable] = None,
        style: Optional[ButtonStyle] = None,
        icon: str = None
    ):
        self.rect = rect
        self.text = text
        self.callback = callback
        self.style = style or ButtonStyle()
        self.icon = icon
        
        self.hovered = False
        self.pressed = False
        self.enabled = True
        
        # Fuente
        pygame.font.init()
        self.font = pygame.font.SysFont('Segoe UI', self.style.font_size, bold=True)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Procesa eventos y retorna True si fue clickeado."""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered and event.button == 1:
                self.pressed = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.hovered and event.button == 1:
                self.pressed = False
                if self.callback:
                    self.callback()
                return True
            self.pressed = False
        
        return False
    
    def draw(self, surface: pygame.Surface):
        """Dibuja el botón."""
        # Determinar color según estado
        if not self.enabled:
            color = Colors.TEXT_MUTED
        elif self.pressed:
            color = self.style.pressed_color
        elif self.hovered:
            color = self.style.hover_color
        else:
            color = self.style.bg_color
        
        # Dibujar fondo con bordes redondeados
        pygame.draw.rect(
            surface, color, self.rect,
            border_radius=self.style.border_radius
        )
        
        # Borde
        pygame.draw.rect(
            surface, Colors.BORDER_LIGHT, self.rect,
            width=1, border_radius=self.style.border_radius
        )
        
        # Texto
        text_surf = self.font.render(self.text, True, self.style.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        
        # Agregar icono si existe
        if self.icon:
            full_text = f"{self.icon} {self.text}"
            text_surf = self.font.render(full_text, True, self.style.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
        
        surface.blit(text_surf, text_rect)


class Panel:
    """
    Panel contenedor con borde y título opcional.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str = None,
        bg_color: Tuple[int, int, int] = Colors.BG_PANEL
    ):
        self.rect = rect
        self.title = title
        self.bg_color = bg_color
        
        pygame.font.init()
        self.title_font = pygame.font.SysFont('Segoe UI', 14, bold=True)
        self.header_height = 30 if title else 0
    
    @property
    def content_rect(self) -> pygame.Rect:
        """Retorna el área de contenido (sin el header)."""
        return pygame.Rect(
            self.rect.x + 5,
            self.rect.y + self.header_height + 5,
            self.rect.width - 10,
            self.rect.height - self.header_height - 10
        )
    
    def draw(self, surface: pygame.Surface):
        """Dibuja el panel."""
        # Sombra
        shadow_rect = self.rect.copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        pygame.draw.rect(surface, (10, 10, 15), shadow_rect, border_radius=10)
        
        # Fondo principal
        pygame.draw.rect(surface, self.bg_color, self.rect, border_radius=10)
        
        # Borde
        pygame.draw.rect(surface, Colors.BORDER, self.rect, width=1, border_radius=10)
        
        # Header con título
        if self.title:
            header_rect = pygame.Rect(
                self.rect.x, self.rect.y,
                self.rect.width, self.header_height
            )
            pygame.draw.rect(
                surface, Colors.BG_CARD, header_rect,
                border_top_left_radius=10, border_top_right_radius=10
            )
            pygame.draw.line(
                surface, Colors.BORDER,
                (self.rect.x, self.rect.y + self.header_height),
                (self.rect.right, self.rect.y + self.header_height)
            )
            
            # Texto del título
            title_surf = self.title_font.render(self.title, True, Colors.PRIMARY)
            surface.blit(title_surf, (self.rect.x + 10, self.rect.y + 7))


class ProgressBar:
    """
    Barra de progreso animada.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        max_value: float = 100,
        color: Tuple[int, int, int] = Colors.PRIMARY
    ):
        self.rect = rect
        self.max_value = max_value
        self.color = color
        self.value = 0
        self.target_value = 0
        self.animation_speed = 0.1
    
    def set_value(self, value: float):
        """Establece el valor objetivo (se anima hacia él)."""
        self.target_value = max(0, min(value, self.max_value))
    
    def update(self):
        """Actualiza la animación."""
        diff = self.target_value - self.value
        self.value += diff * self.animation_speed
    
    def draw(self, surface: pygame.Surface):
        """Dibuja la barra."""
        # Fondo
        pygame.draw.rect(surface, Colors.BG_CARD, self.rect, border_radius=5)
        
        # Barra de progreso
        if self.value > 0:
            fill_width = int((self.value / self.max_value) * self.rect.width)
            fill_rect = pygame.Rect(
                self.rect.x, self.rect.y,
                fill_width, self.rect.height
            )
            pygame.draw.rect(surface, self.color, fill_rect, border_radius=5)
        
        # Borde
        pygame.draw.rect(surface, Colors.BORDER, self.rect, width=1, border_radius=5)


class Chart:
    """
    Gráfico de líneas simple.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        max_points: int = 50,
        color: Tuple[int, int, int] = Colors.PRIMARY,
        title: str = None
    ):
        self.rect = rect
        self.max_points = max_points
        self.color = color
        self.title = title
        self.data: List[float] = []
        
        pygame.font.init()
        self.font = pygame.font.SysFont('Segoe UI', 10)
        self.title_font = pygame.font.SysFont('Segoe UI', 11, bold=True)
    
    def add_point(self, value: float):
        """Agrega un punto de datos."""
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
    
    def clear(self):
        """Limpia los datos."""
        self.data.clear()
    
    def draw(self, surface: pygame.Surface):
        """Dibuja el gráfico."""
        # Fondo
        pygame.draw.rect(surface, Colors.BG_CARD, self.rect, border_radius=5)
        
        # Título
        title_offset = 0
        if self.title:
            title_surf = self.title_font.render(self.title, True, Colors.TEXT_SECONDARY)
            surface.blit(title_surf, (self.rect.x + 5, self.rect.y + 3))
            title_offset = 18
        
        # Área del gráfico
        graph_rect = pygame.Rect(
            self.rect.x + 5,
            self.rect.y + title_offset + 5,
            self.rect.width - 10,
            self.rect.height - title_offset - 10
        )
        
        # Líneas de grid
        for i in range(5):
            y = graph_rect.y + int(i * graph_rect.height / 4)
            pygame.draw.line(
                surface, Colors.BORDER,
                (graph_rect.x, y), (graph_rect.right, y)
            )
        
        # Dibujar línea de datos
        if len(self.data) > 1:
            max_val = max(self.data) if max(self.data) > 0 else 1
            min_val = min(self.data)
            range_val = max_val - min_val if max_val != min_val else 1
            
            points = []
            for i, val in enumerate(self.data):
                x = graph_rect.x + int(i * graph_rect.width / (len(self.data) - 1))
                y = graph_rect.bottom - int((val - min_val) / range_val * graph_rect.height)
                y = max(graph_rect.y, min(y, graph_rect.bottom))
                points.append((x, y))
            
            pygame.draw.lines(surface, self.color, False, points, 2)
            
            # Punto actual
            if points:
                pygame.draw.circle(surface, self.color, points[-1], 4)
        
        # Borde
        pygame.draw.rect(surface, Colors.BORDER, self.rect, width=1, border_radius=5)


class AlertIndicator:
    """
    Indicador visual del nivel de alerta estilo semáforo.
    """
    
    COLORS = {
        'normal': Colors.ALERT_NORMAL,
        'precaución': Colors.ALERT_PRECAUCION,
        'alerta': Colors.ALERT_ALERTA,
        'emergencia': Colors.ALERT_EMERGENCIA
    }
    
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.level = 0.0
        self.category = 'normal'
        self.pulse_phase = 0
        
        pygame.font.init()
        self.font = pygame.font.SysFont('Segoe UI', 16, bold=True)
        self.small_font = pygame.font.SysFont('Segoe UI', 12)
    
    def set_alert(self, level: float, category: str):
        """Establece el nivel y categoría de alerta."""
        self.level = level
        self.category = category
    
    def update(self):
        """Actualiza la animación."""
        self.pulse_phase += 0.1
        if self.pulse_phase > 2 * np.pi:
            self.pulse_phase = 0
    
    def draw(self, surface: pygame.Surface):
        """Dibuja el indicador."""
        color = self.COLORS.get(self.category, Colors.ALERT_NORMAL)
        
        # Efecto de pulso para emergencia
        alpha = 1.0
        if self.category == 'emergencia':
            alpha = 0.7 + 0.3 * np.sin(self.pulse_phase)
        
        # Fondo
        pygame.draw.rect(surface, Colors.BG_CARD, self.rect, border_radius=10)
        
        # Indicador circular grande
        center = (self.rect.centerx, self.rect.y + 50)
        radius = 35
        
        # Glow effect
        for r in range(radius + 10, radius, -2):
            glow_color = tuple(int(c * 0.3) for c in color)
            pygame.draw.circle(surface, glow_color, center, r)
        
        # Círculo principal
        pygame.draw.circle(surface, color, center, radius)
        pygame.draw.circle(surface, Colors.TEXT_PRIMARY, center, radius, 2)
        
        # Porcentaje dentro del círculo
        pct_text = self.font.render(f"{self.level:.0%}", True, Colors.BG_DARK)
        pct_rect = pct_text.get_rect(center=center)
        surface.blit(pct_text, pct_rect)
        
        # Categoría debajo
        cat_text = self.font.render(self.category.upper(), True, color)
        cat_rect = cat_text.get_rect(centerx=self.rect.centerx, y=self.rect.y + 95)
        surface.blit(cat_text, cat_rect)
        
        # Barra de nivel
        bar_rect = pygame.Rect(
            self.rect.x + 10, self.rect.bottom - 25,
            self.rect.width - 20, 15
        )
        pygame.draw.rect(surface, Colors.BG_DARK, bar_rect, border_radius=5)
        
        fill_width = int(self.level * bar_rect.width)
        if fill_width > 0:
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
            pygame.draw.rect(surface, color, fill_rect, border_radius=5)


class Slider:
    """
    Control deslizante para ajustar valores.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        min_val: float = 0,
        max_val: float = 100,
        initial: float = 50,
        label: str = None
    ):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        
        self.dragging = False
        
        pygame.font.init()
        self.font = pygame.font.SysFont('Segoe UI', 11)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Procesa eventos."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value(event.pos[0])
                return True
        
        return False
    
    def _update_value(self, x: int):
        """Actualiza el valor basado en la posición X."""
        relative = (x - self.rect.x) / self.rect.width
        relative = max(0, min(1, relative))
        self.value = self.min_val + relative * (self.max_val - self.min_val)
    
    def draw(self, surface: pygame.Surface):
        """Dibuja el slider."""
        # Label
        if self.label:
            label_text = self.font.render(
                f"{self.label}: {self.value:.1f}",
                True, Colors.TEXT_SECONDARY
            )
            surface.blit(label_text, (self.rect.x, self.rect.y - 15))
        
        # Track
        track_rect = pygame.Rect(
            self.rect.x, self.rect.centery - 3,
            self.rect.width, 6
        )
        pygame.draw.rect(surface, Colors.BG_CARD, track_rect, border_radius=3)
        
        # Filled portion
        relative = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(relative * track_rect.width)
        if fill_width > 0:
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_width, track_rect.height)
            pygame.draw.rect(surface, Colors.PRIMARY, fill_rect, border_radius=3)
        
        # Handle
        handle_x = self.rect.x + int(relative * self.rect.width)
        pygame.draw.circle(surface, Colors.PRIMARY, (handle_x, self.rect.centery), 8)
        pygame.draw.circle(surface, Colors.TEXT_PRIMARY, (handle_x, self.rect.centery), 8, 2)


class StatCard:
    """
    Tarjeta de estadística con valor grande y etiqueta.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        icon: str = None,
        color: Tuple[int, int, int] = Colors.PRIMARY
    ):
        self.rect = rect
        self.label = label
        self.icon = icon
        self.color = color
        self.value = "0"
        
        pygame.font.init()
        self.value_font = pygame.font.SysFont('Segoe UI', 24, bold=True)
        self.label_font = pygame.font.SysFont('Segoe UI', 11)
    
    def set_value(self, value):
        """Establece el valor a mostrar."""
        self.value = str(value)
    
    def draw(self, surface: pygame.Surface):
        """Dibuja la tarjeta."""
        # Fondo
        pygame.draw.rect(surface, Colors.BG_CARD, self.rect, border_radius=8)
        
        # Borde superior con color
        top_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 4)
        pygame.draw.rect(surface, self.color, top_rect, 
                        border_top_left_radius=8, border_top_right_radius=8)
        
        # Valor
        display = f"{self.icon} {self.value}" if self.icon else self.value
        value_surf = self.value_font.render(display, True, self.color)
        value_rect = value_surf.get_rect(centerx=self.rect.centerx, y=self.rect.y + 15)
        surface.blit(value_surf, value_rect)
        
        # Label
        label_surf = self.label_font.render(self.label, True, Colors.TEXT_SECONDARY)
        label_rect = label_surf.get_rect(centerx=self.rect.centerx, y=self.rect.y + 48)
        surface.blit(label_surf, label_rect)
