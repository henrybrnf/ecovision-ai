# Módulo Detector - CNN YOLO
"""
Detección de objetos en tiempo real con YOLOv8.

Este módulo proporciona:
- YOLODetector: Detector de objetos usando YOLOv8
- Detection: Dataclass que representa un objeto detectado
- VideoProcessor: Procesador de video para diferentes fuentes

Example:
    >>> from src.detector import YOLODetector, VideoProcessor
    >>> 
    >>> detector = YOLODetector()
    >>> with VideoProcessor("video.mp4") as video:
    ...     for frame in video:
    ...         detections = detector.detect(frame)
    ...         print(f"Detectados: {len(detections)} objetos")
"""

from .yolo_detector import YOLODetector, Detection
from .video_processor import VideoProcessor, create_test_video
from .tracker import SimpleTracker
from .motion_detector import MotionDetector

__all__ = [
    "YOLODetector",
    "Detection",
    "VideoProcessor",
    "create_test_video"
]
