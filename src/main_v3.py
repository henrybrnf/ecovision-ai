"""
EcoVision AI v3.0 - Interfaz Clara y Profesional

Versión con dashboard mejorado:
- Semáforo grande y visible
- Fuentes claras
- Colores de alerta consistentes
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.detector import YOLODetector, VideoProcessor, create_test_video
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig
from src.visualization.dashboard_v3 import DashboardV3, DashboardConfig


class EcoVisionV3:
    """Aplicación v3.0 con interfaz clara."""
    
    def __init__(self, video_source: str = None, use_webcam: bool = False):
        self.video_source = video_source
        self.use_webcam = use_webcam
        
        self.detector = None
        self.video_processor = None
        self.alert_system = None
        self.simulation = None
        self.dashboard = None
        
        self.running = False
        self.paused = False
        self.speed = 1
        self.frame_count = 0
        
        print("=" * 60)
        print("🎯 EcoVision AI v3.0 - Interfaz Clara")
        print("=" * 60)
    
    def initialize(self):
        """Inicializa componentes."""
        print("\n🔄 Inicializando...")
        
        # Detector
        print("  [1/4] Detector YOLO...")
        self.detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.35,
            classes=[0]
        )
        
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
        self.alert_system = AlertSystem(max_persons=15)
        
        # Ecosistema
        print("  [4/4] Ecosistema evolutivo...")
        self.simulation = Simulation(config=SimulationConfig(
            world_width=600,
            world_height=400,
            agent_count=10,
            steps_per_generation=150,
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
        last_detections = []
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
                        
                        # Detectar
                        last_detections = self.detector.detect(frame)
                        
                        # Evaluar alerta
                        n = len(last_detections)
                        result = self.alert_system.evaluate(
                            person_count=n,
                            movement_speed=min(n / 6, 1.0),
                            zone_density=n / 12
                        )
                        last_alert_level = result.alert_level
                        last_alert_category = result.alert_category
                        
                        # Actualizar ecosistema
                        positions = [d.center for d in last_detections]
                        for _ in range(self.speed):
                            self.simulation.update(positions, last_alert_level)
                
                # Actualizar dashboard
                stats = self.simulation.get_statistics()
                stats['frame_count'] = self.frame_count
                
                self.dashboard.update(
                    frame=last_frame,
                    detections=last_detections,
                    agents=self.simulation.agents,
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
    args = parser.parse_args()
    
    app = EcoVisionV3(video_source=args.video, use_webcam=args.webcam)
    app.run()


if __name__ == "__main__":
    main()
