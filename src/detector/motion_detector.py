import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MotionRegion:
    bbox: Tuple[int, int, int, int] # x1, y1, x2, y2
    area: int
    center: Tuple[int, int]

class MotionDetector:
    """
    Detector de movimiento basado en patrones (Sustracción de Fondo).
    Ideal para detectar actividad donde el Deep Learning falla (vistas aéreas, etc).
    """
    def __init__(self, history=500, varThreshold=32, detectShadows=True): # Threshold aumentado para reducir ruido
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=varThreshold, 
            detectShadows=detectShadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_area = 300 

    def detect(self, frame: np.ndarray) -> List[MotionRegion]:
        """Detecta regiones con movimiento significativo."""
        # Suavizar
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Sustracción
        fgMask = self.backSub.apply(blur)
        
        # Eliminar sombras (valor 127) si detectShadows=True
        # MOG2 marca sombras como gris (127), Foreground como blanco (255)
        _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
        
        # Limpieza (Dilatación reducida a 1 itaración para no inflar la caja)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, self.kernel, iterations=1)
        
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # REFINAMIENTO: Recortar la caja un 10% para ajustarla a la persona
                # y eliminar el "aura" de movimiento
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.05)
                
                motion_regions.append(MotionRegion(
                    bbox=(x + margin_x, y + margin_y, x + w - margin_x, y + h - margin_y),
                    area=int(area),
                    center=(x + w//2, y + h//2)
                ))
                
        return motion_regions
