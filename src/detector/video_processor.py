"""
Procesador de video para captura y procesamiento de frames.

Este módulo maneja la lectura de videos y webcam, proporcionando
una interfaz simple para obtener frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path


class VideoProcessor:
    """
    Procesa video desde archivo o webcam.
    
    Esta clase proporciona una interfaz unificada para leer
    frames de diferentes fuentes de video.
    
    Attributes:
        source: Fuente del video (ruta o índice de webcam)
        cap: Objeto VideoCapture de OpenCV
        fps: Frames por segundo del video
        frame_count: Número total de frames (0 para webcam)
        width: Ancho del video
        height: Alto del video
    
    Example:
        >>> processor = VideoProcessor("video.mp4")
        >>> for frame in processor:
        ...     # Procesar frame
        ...     cv2.imshow("Video", frame)
        >>> processor.release()
    """
    
    def __init__(
        self,
        source: str = "0",
        resize: Optional[Tuple[int, int]] = None
    ):
        """
        Inicializa el procesador de video.
        
        Args:
            source: Ruta al archivo de video o "0" para webcam
            resize: Tupla (width, height) para redimensionar frames
        """
        self.source = source
        self.resize = resize
        self._frame_idx = 0
        
        # Determinar si es webcam o archivo
        if source.isdigit():
            self.is_webcam = True
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.is_webcam = False
            if not Path(source).exists():
                raise FileNotFoundError(f"Video no encontrado: {source}")
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        
        # Obtener propiedades del video
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video abierto: {source}")
        print(f"   Resolución: {self.width}x{self.height}")
        print(f"   FPS: {self.fps}")
        if not self.is_webcam:
            print(f"   Frames totales: {self.frame_count}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el siguiente frame.
        
        Returns:
            Tupla (success, frame) donde success indica si se leyó correctamente
        """
        ret, frame = self.cap.read()
        
        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)
        
        if ret:
            self._frame_idx += 1
        
        return ret, frame
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterador para procesar todos los frames."""
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
    
    def __len__(self) -> int:
        """Retorna el número de frames (0 para webcam)."""
        return self.frame_count if not self.is_webcam else 0
    
    @property
    def current_frame(self) -> int:
        """Índice del frame actual."""
        return self._frame_idx
    
    @property
    def progress(self) -> float:
        """Progreso del video (0.0 - 1.0)."""
        if self.is_webcam or self.frame_count == 0:
            return 0.0
        return self._frame_idx / self.frame_count
    
    def seek(self, frame_idx: int) -> bool:
        """
        Salta a un frame específico.
        
        Args:
            frame_idx: Índice del frame destino
        
        Returns:
            True si el salto fue exitoso
        """
        if self.is_webcam:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._frame_idx = frame_idx
        return True
    
    def reset(self) -> None:
        """Reinicia el video al principio."""
        self.seek(0)
    
    def release(self) -> None:
        """Libera los recursos del video."""
        if self.cap:
            self.cap.release()
            print("📹 Video liberado")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
    
    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Obtiene un frame específico sin avanzar el contador.
        
        Args:
            frame_idx: Índice del frame a obtener
        
        Returns:
            El frame en la posición indicada o None si falla
        """
        current = self._frame_idx
        self.seek(frame_idx)
        ret, frame = self.read()
        self.seek(current)
        return frame if ret else None


def create_test_video(
    output_path: str = "data/videos/test.mp4",
    duration: int = 5,
    fps: int = 30,
    size: Tuple[int, int] = (640, 480)
) -> str:
    """
    Crea un video de prueba con formas en movimiento.
    
    Args:
        output_path: Ruta de salida
        duration: Duración en segundos
        fps: Frames por segundo
        size: Tamaño del video (width, height)
    
    Returns:
        Ruta del video creado
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Crear frame con fondo
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # Gris oscuro
        
        # Dibujar círculos en movimiento
        t = i / fps
        x = int(size[0] / 2 + 200 * np.cos(t * 2))
        y = int(size[1] / 2 + 100 * np.sin(t * 2))
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        
        # Dibujar rectángulo
        x2 = int(size[0] / 2 + 150 * np.sin(t * 3))
        y2 = int(size[1] / 2 + 80 * np.cos(t * 3))
        cv2.rectangle(frame, (x2 - 20, y2 - 20), (x2 + 20, y2 + 20), (255, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Video de prueba creado: {output_path}")
    return output_path


# Para pruebas rápidas
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Test del Procesador de Video")
    print("=" * 50)
    
    # Crear video de prueba
    test_video = create_test_video("data/videos/test.mp4", duration=2)
    
    # Probar el procesador
    with VideoProcessor(test_video) as processor:
        print(f"\n📊 Propiedades del video:")
        print(f"   FPS: {processor.fps}")
        print(f"   Frames: {processor.frame_count}")
        print(f"   Resolución: {processor.width}x{processor.height}")
        
        # Leer algunos frames
        frame_count = 0
        for frame in processor:
            frame_count += 1
            if frame_count >= 10:
                break
        
        print(f"\n✅ Leídos {frame_count} frames correctamente!")
    
    print("\n✅ Test completado exitosamente!")
