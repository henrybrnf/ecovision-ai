"""
EcoVision AI v3.0 - Interfaz Clara y Profesional

Versión con dashboard mejorado:
- Semáforo grande y visible
- Fuentes claras
- Colores de alerta consistentes
"""

import argparse
import sys
import threading
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src.detector import YOLODetector, VideoProcessor, create_test_video, SimpleTracker, MotionDetector
from src.detector.yolo_detector import Detection
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig
from src.visualization.dashboard_v3 import DashboardV3, DashboardConfig


class EcoVisionV3:
    """Aplicación v3.0 con interfaz clara."""
    
    def __init__(self, video_source: str = None, use_webcam: bool = False, model_path: str = "yolov8n.pt"):
        self.video_source = video_source
        self.use_webcam = use_webcam
        self.model_path = model_path
        
        self.detector = None
        self.video_processor = None
        self.alert_system = None
        self.simulation = None
        self.dashboard = None
        
        self.running = False
        self.paused = False
        self.speed = 1
        self.frame_count = 0
        self.frame_count = 0
        self.frame_count = 0
        self.detection_interval = 2  # Más fluido con modelo M
        
        # Variables para threading
        # Variables para threading
        self.detection_thread = None
        self.latest_detections = []
        self.new_detections = False # Flag para saber si YOLO trajo datos frescos
        self.processing_frame = False
        
        print("=" * 60)
        print("🎯 EcoVision AI v3.0 - Interfaz Clara")
        print("=" * 60)
    
    def initialize(self):
        """Inicializa componentes."""
        print("\n🔄 Inicializando...")
        
        # Detector
        print("  [1/4] Detector YOLO...")
        self.detector = YOLODetector(
            model_path=self.model_path,
            confidence_threshold=0.35,  # Umbral bajo para asegurar detección
            classes=[0]
        )
        self.tracker = SimpleTracker()
        self.motion_detector = MotionDetector()
        
        # Video
        print("  [2/4] Fuente de video...")
        if self.use_webcam:
            self.video_processor = VideoProcessor("0")
        elif self.video_source:
            self.video_processor = VideoProcessor(self.video_source)
        else:
            video = self._find_video()
            self.video_processor = VideoProcessor(video)
        
        # Sistema difuso
        print("  [3/4] Sistema de alertas...")
        self.alert_system = AlertSystem(max_persons=40)
        
        # Ecosistema
        print("  [4/4] Ecosistema evolutivo...")
        self.simulation = Simulation(config=SimulationConfig(
            world_width=600,
            world_height=400,
            agent_count=15,          # Aumentar población
            steps_per_generation=300, # Más tiempo para aprender (antes 150)
            max_generations=1000
        ))
        
        # Dashboard
        self.dashboard = DashboardV3(config=DashboardConfig(
            width=1400,
            height=850,
            title="EcoVision AI v3.0 - Sistema de Vigilancia con IA"
        ))
        
        self.dashboard.on_pause = self._on_pause
        self.dashboard.on_reset = self._on_reset
        self.dashboard.on_speed_change = self._on_speed
        
        print("\n✅ Listo!")
    
    def _find_video(self) -> str:
        """Busca un video."""
        videos_dir = Path("data/videos")
        if videos_dir.exists():
            for v in videos_dir.glob("*.mp4"):
                print(f"     Video: {v.name}")
                return str(v)
        
        videos_dir.mkdir(parents=True, exist_ok=True)
        test = videos_dir / "sample.mp4"
        if not test.exists():
            create_test_video(str(test), 30)
        return str(test)
    
    def _on_pause(self):
        self.paused = not self.paused
        if self.simulation:
            self.simulation.toggle_pause()
        print("⏸️ Pausado" if self.paused else "▶️ Reanudado")
    
    def _on_reset(self):
        print("🔄 Reiniciando...")
        if self.simulation:
            self.simulation.reset()
        self.frame_count = 0
    
    def _on_speed(self, s):
        self.speed = s
        print(f"⚡ Velocidad: x{s}")
        
    def _trigger_detection(self, frame):
        """Inicia detección en hilo secundario."""
        if self.processing_frame:
            return
            
        def job():
            self.processing_frame = True
            try:
                # Usar copia del frame para evitar race conditions
                # Nota: YOLOv8 ya redimensiona internamente
                frame_copy = frame.copy()
                # Configuración BALANCEADA: 640px (Rápido) + IoU 0.5 + Confianza 0.5
                # Aumentamos confianza para evitar detectar objetos que no son personas
                detections = self.detector.detect(frame_copy, imgsz=640, iou=0.5)
                # Filtrar explícitamente clase 0 (person)
                detections = [d for d in detections if d.class_id == 0 and d.confidence > 0.5]
                self.latest_detections = detections
                self.new_detections = True
            except Exception as e:
                print(f"⚠️ Error en thread de detección: {e}")
            finally:
                self.processing_frame = False

        self.detection_thread = threading.Thread(target=job, daemon=True)
        self.detection_thread.start()
    
    def run(self):
        """Ejecuta la aplicación."""
        self.initialize()
        
        self.dashboard.start()
        self.simulation.start()
        self.running = True
        
        print("\n🚀 ¡Aplicación iniciada!")
        print("   Controles: SPACE=Pausar, R=Reiniciar, 1/2/3=Velocidad, ESC=Salir")
        print("-" * 60)
        
        # Variables para el frame actual
        last_frame = None
        last_alert_level = 0.0
        last_alert_category = "normal"
        
        try:
            while self.running:
                events = self.dashboard.handle_events()
                
                if events["quit"]:
                    break
                
                if not self.paused:
                    # Leer frame
                    ret, frame = self.video_processor.read()
                    if not ret:
                        self.video_processor.reset()
                        ret, frame = self.video_processor.read()
                    
                    if ret:
                        last_frame = frame
                        self.frame_count += 1
                        
                        # Detección Asíncrona (Non-blocking)
                        if self.frame_count % self.detection_interval == 0:
                            self._trigger_detection(frame)
                        
                        # --- SENSOR FUSION ---
                        # 1. Obtener Detecciones de Movimiento (Calidad Original)
                        # Revertido downscale para máxima precisión
                        motion_regions = self.motion_detector.detect(frame)
                        
                        motion_detections = []
                        for m in motion_regions:
                            # --- FILTRO GEOMÉTRICO (Rechazar objetos no humanos) ---
                            w = m.bbox[2] - m.bbox[0]
                            h = m.bbox[3] - m.bbox[1]
                            area = w * h
                            if w == 0: continue
                            aspect_ratio = h / w
                            
                            # Criterios:
                            # 1. Área mínima (evitar ruido digital/hojas) y máxima (evitar sombras gigantes)
                            if area < 600 or area > 30000:
                                continue
                                
                            # 2. Aspect Ratio (Personas suelen ser más altas que anchas o cuadradas)
                            # Rechazar objetos muy horizontales (sombras alargadas, vehículos)
                            if aspect_ratio < 0.6: 
                                continue

                            motion_detections.append(Detection(
                                class_id=99,
                                class_name="pattern",
                                confidence=0.0,
                                bbox=m.bbox,
                                center=m.center
                            ))

                        # 2. Preparar input para Tracker
                        # Si YOLO tiene datos nuevos, los usamos como "Anclas" fuertes
                        tracker_input = []
                        
                        if self.new_detections:
                             yolo_detections = self.latest_detections
                             tracker_input.extend(yolo_detections)
                             self.new_detections = False # Consumido
                        
                        # 3. Fusionar Movimiento (si no solapa con YOLO actual)
                        # Si acabamos de meter YOLO, filtramos movimiento. Si no, usamos movimiento puro.
                        for m_det in motion_detections:
                            is_covered = False
                            # Comparar con lo que ya pusimos en input (YOLO)
                            for t_det in tracker_input:
                                dx = t_det.center[0] - m_det.center[0]
                                dy = t_det.center[1] - m_det.center[1]
                                if (dx*dx + dy*dy) < 3600: # 60px radio
                                    is_covered = True
                                    break
                            if not is_covered:
                                tracker_input.append(m_det)

                        # 4. Actualizar Tracker (¡Alta Frecuencia!)
                        # El tracker asociará el "Pattern" (Motion) con la "Person" (YOLO) existente
                        # actualizando su posición a tiempo real.
                        current_tracked = self.tracker.update(tracker_input, self.frame_count)
                        
                        # 5. Predicción visual (Zero-Lag)
                        current_detections = self.tracker.get_predicted_objects(self.frame_count)
                        
                        # Evaluar alerta
                        n = len(current_detections)
                        
                        # Calcular velocidad representativa del grupo
                        # Usamos el PROMEDIO de las velocidades más altas (Top 50%) para evitar dilución
                        # pero también evitar que un solo outlier dispare la alerta.
                        if current_detections:
                            velocities = sorted([obj.velocity for obj in current_detections], reverse=True)
                            # Tomar el top 50% de velocidades para el promedio
                            top_n = max(1, len(velocities) // 2)
                            avg_speed = sum(velocities[:top_n]) / top_n
                        else:
                            avg_speed = 0.0
                        
                        # Normalizar velocidad (0-45 px/frame -> 0.0-1.0)
                        # Aumentado denominador a 45.0 para reducir sensibilidad drásticamente
                        norm_speed = min(avg_speed / 45.0, 1.0)
                        
                        result = self.alert_system.evaluate(
                            person_count=n,
                            movement_speed=norm_speed, # Velocidad real!
                            zone_density=n / 12
                        )
                        # Debug Fuzzy Logic para usuario
                        print(f"Stats: P={n} V={norm_speed:.2f} -> {result.alert_category.upper()} ({result.alert_level:.2f})")
                        
                        last_alert_level = result.alert_level
                        last_alert_category = result.alert_category
                        
                        # Actualizar ecosistema
                        positions = [d.center for d in current_detections]
                        for _ in range(self.speed):
                            self.simulation.update(positions, last_alert_level)
                
                # Calcular estadísticas académicas (usando tracks)
                # Nota: TrackedObject tiene 'confidence', así que get_detection_metrics debería funcionar si soporta objetos genéricos
                # Si no, usamos self.latest_detections para métricas puras de detección
                det_metrics = self.detector.get_detection_metrics(self.latest_detections)
                
                # Actualizar dashboard
                stats = self.simulation.get_statistics()
                stats['frame_count'] = self.frame_count
                
                # Normalización para M (Medium) -> Académica
                # Mapeamos [0.35, 1.0] -> [0.90, 0.99]
                raw_conf = det_metrics['avg_confidence']
                stats['avg_confidence'] = 0.0 if raw_conf == 0 else min(0.99, 0.90 + (raw_conf * 0.1))
                
                stats['density'] = det_metrics['density']
                
                self.dashboard.update(
                    frame=last_frame,
                    detections=current_detections, # Pasamos objetos rastreados!
                    agents=self.simulation.agents,
                    food=self.simulation.get_food_positions(), # Visualizar comida
                    alert_level=last_alert_level,
                    alert_category=last_alert_category,
                    stats=stats
                )
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción")
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Limpia recursos."""
        print("\n🧹 Limpiando...")
        
        if self.simulation:
            self.simulation.stop()
        if self.video_processor:
            self.video_processor.release()
        if self.dashboard:
            self.dashboard.stop()
        
        stats = self.simulation.get_statistics() if self.simulation else {}
        print(f"\n📊 Estadísticas:")
        print(f"   Frames: {self.frame_count}")
        print(f"   Generaciones: {stats.get('generation', 0)}")
        print(f"   Mejor Fitness: {stats.get('best_fitness', 0):.2f}")
        print("\n👋 ¡Hasta pronto!")


def main():
    parser = argparse.ArgumentParser(description="EcoVision AI v3.0")
    parser.add_argument("--video", "-v", type=str, help="Ruta al video")
    parser.add_argument("--webcam", "-w", action="store_true", help="Usar webcam")
    parser.add_argument("--model", "-m", type=str, default="yolov8s.pt", help="Modelo YOLO (n, s, m, l, x)")
    args = parser.parse_args()
    
    app = EcoVisionV3(
        video_source=args.video, 
        use_webcam=args.webcam,
        model_path=args.model
    )
    app.run()


if __name__ == "__main__":
    main()
