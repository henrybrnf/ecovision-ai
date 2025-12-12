# Módulo Visualization
"""
Dashboard y visualización del sistema EcoVision AI.

Este módulo proporciona:
- Dashboard: Interfaz principal que integra todo
- EcosystemRenderer: Renderizado del ecosistema 2D
- VideoRenderer: Renderizado de video con detecciones

Example:
    >>> from src.visualization import Dashboard
    >>> 
    >>> dashboard = Dashboard()
    >>> dashboard.start()
    >>> dashboard.update(frame, detections, agents, alert_level)
    >>> dashboard.stop()
"""

from .renderer import EcosystemRenderer, VideoRenderer, RenderConfig
from .dashboard import Dashboard

__all__ = [
    "Dashboard",
    "EcosystemRenderer",
    "VideoRenderer",
    "RenderConfig"
]
