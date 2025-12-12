# Módulo Visualization
"""
Dashboard y visualización del sistema EcoVision AI.

Este módulo proporciona:
- DashboardV2: Interfaz principal mejorada con botones
- EcosystemRenderer: Renderizado del ecosistema 2D
- VideoRenderer: Renderizado de video con detecciones
- UI Components: Botones, paneles, gráficos

Example:
    >>> from src.visualization import DashboardV2
    >>> 
    >>> dashboard = DashboardV2()
    >>> dashboard.start()
    >>> dashboard.update(frame, detections, agents, alert_level)
    >>> dashboard.stop()
"""

from .renderer import EcosystemRenderer, VideoRenderer, RenderConfig
from .dashboard import Dashboard
from .dashboard_v2 import DashboardV2, DashboardConfig
from .ui_components import (
    Colors, Button, Panel, Chart, AlertIndicator, 
    StatCard, ProgressBar, Slider, ButtonStyle
)

__all__ = [
    # Dashboards
    "Dashboard",
    "DashboardV2",
    "DashboardConfig",
    # Renderers
    "EcosystemRenderer",
    "VideoRenderer",
    "RenderConfig",
    # UI Components
    "Colors",
    "Button",
    "Panel",
    "Chart",
    "AlertIndicator",
    "StatCard",
    "ProgressBar",
    "Slider",
    "ButtonStyle"
]
