"""
EcoVision AI v2.0 - Aplicación Principal Mejorada

Sistema Inteligente de Vigilancia con:
- Dashboard interactivo con botones y gráficos
- Detección de personas en tiempo real
- Ecosistema evolutivo visual
- Alertas con lógica difusa

Uso:
    python src/main_v2.py                    # Demo con video de prueba
    python src/main_v2.py --webcam           # Usar webcam
    python src/main_v2.py --video path.mp4   # Video específico
"""

import argparse
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.detector import YOLODetector, VideoProcessor, create_test_video
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig
from src.visualization import DashboardV2, DashboardConfig


class EcoVisionAppV2:
    """
    Aplicación principal de EcoVision AI v2.0.
    
    Mejoras:
    - Dashboard interactivo con botones
    - Gráficos en tiempo real
    - Mejor visualización del ecosistema
    - Control de velocidad
    """
    
    def __init__(
        self,
        video_source: str = None,
        use_webcam: bool = False
    ):
        self.video_source = video_source
        self.use_webcam = use_webcam
        
        # Componentes
        self.detector = None
        self.video_processor = None
        self.alert_system = None
        self.simulation = None
        self.dashboard = None
        
        # Estado
        self.running = False
        self.paused = False
        self.speed_multiplier = 1
        self.frame_count = 0
        
        self._print_header()
    
    def _print_header(self):
        """Imprime el header de la aplicación."""
        print("=" * 60)
        print("🎯 EcoVision AI v2.0")
        print("   Sistema Inteligente de Vigilancia Mejorado")
        print("=" * 60)
    
    def initialize(self):
        """Inicializa todos los componentes."""
        print("\n🔄 Inicializando componentes...")
        
        # 1. Detector YOLO
        print("\n📦 [1/4] Detector YOLO")
        self.detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.35,
            classes=[0]
        )
        
        # 2. Fuente de video
        if self.use_webcam:
            print("📹 Usando webcam...")
            self.video_processor = VideoProcessor("0")
        elif self.video_source:
            print(f"📹 Usando video: {self.video_source}")
            self.video_processor = VideoProcessor(self.video_source)
        else:
            # Buscar video existente o crear uno de prueba
            test_video = self._find_or_create_test_video()
            self.video_processor = VideoProcessor(test_video)
        
        # 3. Sistema de Lógica Difusa
        print("\n📦 [2/4] Sistema de Lógica Difusa")
        self.alert_system = AlertSystem(max_persons=20)
        
        # 4. Ecosistema
        print("\n📦 [3/4] Ecosistema Evolutivo")
        sim_config = SimulationConfig(
            world_width=600,
            world_height=400,
            agent_count=12,
            steps_per_generation=200,
            max_generations=1000
        )
        self.simulation = Simulation(config=sim_config)
        
        # 5. Dashboard
        print("\n📦 [4/4] Dashboard Interactivo")
        dashboard_config = DashboardConfig(
            width=1400,
            height=800,
            fps=30,
            title="EcoVision AI v2.0 - Sistema de Vigilancia Inteligente",
            resizable=True
        )
        self.dashboard = DashboardV2(config=dashboard_config)
        
        # Conectar callbacks
        self.dashboard.on_pause = self._toggle_pause
        self.dashboard.on_reset = self._reset_simulation
        self.dashboard.on_speed_change = self._change_speed
        
        print("\n✅ Todos los componentes inicializados")
    
    def _find_or_create_test_video(self) -> str:
        """Busca un video existente o crea uno de prueba."""
        videos_dir = Path("data/videos")
        
        # Buscar videos existentes
        if videos_dir.exists():
            for video_file in videos_dir.glob("*.mp4"):
                print(f"📹 Usando video existente: {video_file.name}")
                return str(video_file)
        
        # Crear video de prueba
        videos_dir.mkdir(parents=True, exist_ok=True)
        test_video = videos_dir / "sample.mp4"
        
        if not test_video.exists():
            print("📹 Creando video de prueba...")
            create_test_video(str(test_video), duration=30)
        
        return str(test_video)
    
    def _toggle_pause(self):
        """Alterna el estado de pausa."""
        self.paused = not self.paused
        status = "⏸️ Pausado" if self.paused else "▶️ Reanudado"
        print(f"\n{status}")
        
        if self.simulation:
            self.simulation.toggle_pause()
    
    def _reset_simulation(self):
        """Reinicia la simulación."""
        print("\n🔄 Reiniciando simulación...")
        if self.simulation:
            self.simulation.reset()
        self.frame_count = 0
    
    def _change_speed(self, speed: int):
        """Cambia la velocidad de simulación."""
        self.speed_multiplier = speed
        print(f"\n⚡ Velocidad: x{speed}")
    
    def run(self):
        """Ejecuta el loop principal."""
        self.initialize()
        
        self.dashboard.start()
        self.simulation.start()
        
        self.running = True
        print("\n🚀 ¡Aplicación iniciada!")
        print("   Click en botones o usa teclas: SPACE, R, ESC")
        print("-" * 60)
        
        try:
            while self.running:
                # Eventos
                events = self.dashboard.handle_events()
                
                if events["quit"]:
                    self.running = False
                    break
                
                # Saltar si está pausado
                if self.paused:
                    self.dashboard.update(
                        frame=self.last_frame if hasattr(self, 'last_frame') else None,
                        detections=self.last_detections if hasattr(self, 'last_detections') else [],
                        agents=self.simulation.agents,
                        alert_level=self.last_alert_level if hasattr(self, 'last_alert_level') else 0,
                        alert_category=self.last_alert_category if hasattr(self, 'last_alert_category') else "normal",
                        stats=self._get_stats()
                    )
                    continue
                
                # Leer frame
                ret, frame = self.video_processor.read()
                if not ret:
                    self.video_processor.reset()
                    ret, frame = self.video_processor.read()
                    if not ret:
                        continue
                
                self.last_frame = frame
                
                # Detectar personas
                detections = self.detector.detect(frame)
                self.last_detections = detections
                
                # Evaluar alerta
                person_count = len(detections)
                movement_speed = min(person_count / 8, 1.0)
                
                result = self.alert_system.evaluate(
                    person_count=person_count,
                    movement_speed=movement_speed,
                    zone_density=person_count / 15
                )
                
                self.last_alert_level = result.alert_level
                self.last_alert_category = result.alert_category
                
                # Obtener posiciones de detecciones
                det_positions = [d.center for d in detections]
                
                # Actualizar ecosistema (múltiples pasos según velocidad)
                for _ in range(self.speed_multiplier):
                    self.simulation.update(det_positions, result.alert_level)
                
                self.frame_count += 1
                
                # Actualizar dashboard
                self.dashboard.update(
                    frame=frame,
                    detections=detections,
                    agents=self.simulation.agents,
                    alert_level=result.alert_level,
                    alert_category=result.alert_category,
                    stats=self._get_stats()
                )
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción de teclado")
        
        finally:
            self.cleanup()
    
    def _get_stats(self) -> dict:
        """Obtiene las estadísticas actuales."""
        sim_stats = self.simulation.get_statistics() if self.simulation else {}
        
        return {
            'frame_count': self.frame_count,
            'generation': sim_stats.get('generation', 0),
            'step': sim_stats.get('step', 0),
            'steps_per_gen': sim_stats.get('steps_per_gen', 100),
            'best_fitness': sim_stats.get('best_fitness', 0),
            'avg_fitness': sim_stats.get('avg_fitness', 0),
            'agent_count': sim_stats.get('agent_count', 0),
            'paused': self.paused
        }
    
    def cleanup(self):
        """Limpia recursos."""
        print("\n🧹 Limpiando recursos...")
        
        if self.simulation:
            self.simulation.stop()
        
        if self.video_processor:
            self.video_processor.release()
        
        if self.dashboard:
            self.dashboard.stop()
        
        print("\n📊 Estadísticas finales:")
        print(f"   Frames procesados: {self.frame_count}")
        
        if self.simulation:
            stats = self.simulation.get_statistics()
            print(f"   Generaciones: {stats.get('generation', 0)}")
            print(f"   Mejor fitness: {stats.get('best_fitness', 0):.2f}")
        
        print("\n👋 ¡Hasta pronto!")


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="EcoVision AI v2.0 - Sistema de Vigilancia Inteligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python src/main_v2.py                     Demo con video automático
  python src/main_v2.py --video video.mp4   Usar video específico
  python src/main_v2.py --webcam            Usar webcam
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Ruta al archivo de video"
    )
    
    parser.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Usar webcam"
    )
    
    args = parser.parse_args()
    
    app = EcoVisionAppV2(
        video_source=args.video,
        use_webcam=args.webcam
    )
    
    app.run()


if __name__ == "__main__":
    main()
