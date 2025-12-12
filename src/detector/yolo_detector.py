"""
Detector de objetos usando YOLOv8.

Este módulo implementa la detección de personas y objetos en tiempo real
utilizando el modelo YOLOv8 de Ultralytics.
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Representa un objeto detectado en la imagen."""
    class_id: int           # ID de la clase detectada
    class_name: str         # Nombre de la clase (ej: "person")
    confidence: float       # Confianza de la detección (0.0 - 1.0)
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    center: Tuple[int, int]  # Centro del objeto (cx, cy)
    
    @property
    def width(self) -> int:
        """Ancho del bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Alto del bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        """Área del bounding box."""
        return self.width * self.height


class YOLODetector:
    """
    Detector de objetos usando YOLOv8.
    
    Esta clase encapsula el modelo YOLO y proporciona una interfaz
    simple para detectar objetos en imágenes.
    
    Attributes:
        model: Modelo YOLOv8 cargado
        confidence_threshold: Umbral mínimo de confianza
        classes: Lista de clases a detectar (None = todas)
    
    Example:
        >>> detector = YOLODetector()
        >>> detections = detector.detect(frame)
        >>> for d in detections:
        ...     print(f"{d.class_name}: {d.confidence:.2f}")
    """
    
    # Mapeo de clases COCO relevantes
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        16: "dog",
        17: "cat",
        # ... se pueden agregar más
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        """
        Inicializa el detector YOLO.
        
        Args:
            model_path: Ruta al modelo YOLO (se descarga automáticamente)
            confidence_threshold: Umbral de confianza (0.0 - 1.0)
            classes: Lista de IDs de clases a detectar (0=person, etc.)
            device: Dispositivo a usar ("cpu" o "cuda")
        """
        print(f"🔄 Cargando modelo YOLO: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes = classes if classes else [0]  # Por defecto solo personas
        self.device = device
        print(f"✅ Modelo YOLO cargado en {device}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta objetos en un frame.
        
        Args:
            frame: Imagen en formato numpy array (BGR o RGB)
        
        Returns:
            Lista de objetos Detection encontrados
        """
        # Ejecutar inferencia
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False  # Silenciar output
        )
        
        detections = []
        
        # Procesar resultados
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                
                # Obtener confianza y clase
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Obtener nombre de clase
                class_name = self.model.names.get(class_id, f"class_{class_id}")
                
                # Calcular centro
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy)
                )
                
                detections.append(detection)
        
        return detections
    
    def detect_and_draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detecta objetos y dibuja los bounding boxes en el frame.
        
        Args:
            frame: Imagen de entrada
            color: Color de los bounding boxes (BGR)
            thickness: Grosor de las líneas
        
        Returns:
            Tuple de (frame con dibujos, lista de detecciones)
        """
        import cv2
        
        detections = self.detect(frame)
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Dibujar rectángulo
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Dibujar etiqueta
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Fondo de la etiqueta
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # Dibujar centro
            cv2.circle(annotated_frame, det.center, 5, (0, 0, 255), -1)
        
        return annotated_frame, detections
    
    def get_person_count(self, frame: np.ndarray) -> int:
        """Cuenta el número de personas en el frame."""
        detections = self.detect(frame)
        return sum(1 for d in detections if d.class_name == "person")
    
    def get_detection_metrics(self, detections: List[Detection]) -> dict:
        """
        Calcula métricas de las detecciones.
        
        Returns:
            Diccionario con métricas: count, avg_confidence, positions, etc.
        """
        if not detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "positions": [],
                "avg_area": 0,
                "density": 0.0
            }
        
        positions = [d.center for d in detections]
        areas = [d.area for d in detections]
        confidences = [d.confidence for d in detections]
        
        return {
            "count": len(detections),
            "avg_confidence": np.mean(confidences),
            "positions": positions,
            "avg_area": np.mean(areas),
            "density": len(detections) / 1000  # Densidad relativa
        }


# Para pruebas rápidas
if __name__ == "__main__":
    import cv2
    
    print("=" * 50)
    print("🧪 Test del Detector YOLO")
    print("=" * 50)
    
    # Crear detector
    detector = YOLODetector(confidence_threshold=0.5)
    
    # Crear imagen de prueba (negro)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Detectar
    detections = detector.detect(test_frame)
    print(f"\n📊 Detecciones en imagen de prueba: {len(detections)}")
    
    print("\n✅ Test completado exitosamente!")
