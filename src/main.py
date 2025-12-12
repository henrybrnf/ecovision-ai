"""
EcoVision AI - Aplicación Principal

Sistema Inteligente de Vigilancia con Detección de Objetos y Agentes Evolutivos.

Este script integra todos los módulos:
- Detector YOLO para detectar personas
- Sistema de Lógica Difusa para evaluar alertas
- Ecosistema de agentes evolutivos
- Dashboard de visualización

Uso:
    python src/main.py                    # Demo con video de prueba
    python src/main.py --mode webcam      # Usar webcam
    python src/main.py --video path.mp4   # Video específico
    python src/main.py --ecosystem-only   # Solo ecosistema (sin video)
"""

import argparse
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Importar módulos del proyecto
from src.detector import YOLODetector, VideoProcessor, create_test_video
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig
from src.visualization import Dashboard


class EcoVisionApp:
    """
    Aplicación principal de EcoVision AI.
    
    Coordina todos los componentes del sistema.
    """
    
    def __init__(
        self,
        video_source: str = None,
        use_webcam: bool = False,
        ecosystem_only: bool = False
    ):
        """
        Inicializa la aplicación.
        
        Args:
            video_source: Ruta al archivo de video
            use_webcam: Si usar webcam en lugar de video
            ecosystem_only: Si ejecutar solo el ecosistema sin video
        """
        self.video_source = video_source
        self.use_webcam = use_webcam
        self.ecosystem_only = ecosystem_only
        
        # Componentes
        self.detector = None
        self.video_processor = None
        self.alert_system = None
        self.simulation = None
        self.dashboard = None
        
        # Estado
        self.running = False
        self.frame_count = 0
        
        print("=" * 60)
        print("🎯 EcoVision AI - Sistema de Vigilancia Inteligente")
        print("=" * 60)
    
    def initialize(self):
        """Inicializa todos los componentes."""
        print("\n🔄 Inicializando componentes...")
        
        # 1. Inicializar Detector YOLO (si no es ecosystem_only)
        if not self.ecosystem_only:
            print("\n📦 Módulo 1: Detector YOLO")
            self.detector = YOLODetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                classes=[0]  # Solo personas
            )
            
            # Inicializar fuente de video
            if self.use_webcam:
                print("📹 Usando webcam...")
                self.video_processor = VideoProcessor("0")
            elif self.video_source:
                print(f"📹 Usando video: {self.video_source}")
                self.video_processor = VideoProcessor(self.video_source)
            else:
                # Crear video de prueba si no existe
                test_video = Path("data/videos/sample.mp4")
                if not test_video.exists():
                    print("📹 Creando video de prueba...")
                    create_test_video(str(test_video), duration=30)
                self.video_processor = VideoProcessor(str(test_video))
        
        # 2. Inicializar Sistema de Lógica Difusa
        print("\n📦 Módulo 2: Sistema de Lógica Difusa")
        self.alert_system = AlertSystem(max_persons=20)
        
        # 3. Inicializar Ecosistema
        print("\n📦 Módulo 3: Ecosistema Evolutivo")
        sim_config = SimulationConfig(
            world_width=590,  # Tamaño del panel del ecosistema
            world_height=400,
            agent_count=15,
            steps_per_generation=300,
            max_generations=1000
        )
        self.simulation = Simulation(config=sim_config)
        
        # 4. Inicializar Dashboard
        print("\n📦 Módulo 4: Dashboard de Visualización")
        self.dashboard = Dashboard(
            width=1200,
            height=580,
            title="EcoVision AI - Sistema de Vigilancia Inteligente"
        )
        
        print("\n✅ Todos los componentes inicializados")
    
    def run(self):
        """Ejecuta el loop principal de la aplicación."""
        # Inicializar
        self.initialize()
        
        # Iniciar componentes
        self.dashboard.start()
        self.simulation.start()
        
        self.running = True
        print("\n🚀 Aplicación iniciada!")
        print("   Controles: SPACE=Pausar, R=Reiniciar, ESC=Salir")
        print("-" * 60)
        
        try:
            while self.running:
                # Procesar eventos
                events = self.dashboard.handle_events()
                
                if events["quit"]:
                    self.running = False
                    break
                
                if events["pause"]:
                    self.simulation.toggle_pause()
                
                if events["reset"]:
                    self.simulation.reset()
                
                # Obtener frame y detectar (si hay video)
                frame = None
                detections = []
                detection_positions = []
                
                if not self.ecosystem_only and self.video_processor:
                    ret, frame = self.video_processor.read()
                    
                    if not ret:
                        # Reiniciar video si llega al final
                        self.video_processor.reset()
                        ret, frame = self.video_processor.read()
                    
                    if ret and frame is not None:
                        # Detectar objetos
                        detections = self.detector.detect(frame)
                        
                        # Obtener posiciones de detecciones
                        detection_positions = [d.center for d in detections]
                
                # Evaluar nivel de alerta
                person_count = len(detections)
                
                # Calcular velocidad de movimiento (simplificado)
                movement_speed = min(person_count / 10, 1.0)
                
                # Evaluar con sistema difuso
                fuzzy_result = self.alert_system.evaluate(
                    person_count=person_count,
                    movement_speed=movement_speed,
                    zone_density=person_count / 20
                )
                
                # Actualizar ecosistema
                self.simulation.update(
                    detected_objects=detection_positions,
                    alert_level=fuzzy_result.alert_level
                )
                
                # Obtener estadísticas
                stats = self.simulation.get_statistics()
                
                # Actualizar dashboard
                self.dashboard.update(
                    video_frame=frame,
                    detections=detections,
                    agents=self.simulation.agents,
                    alert_level=fuzzy_result.alert_level,
                    alert_category=fuzzy_result.alert_category,
                    stats=stats
                )
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción de teclado detectada")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia recursos y cierra la aplicación."""
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
        description="EcoVision AI - Sistema de Vigilancia Inteligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python src/main.py                     Demo con video de prueba
  python src/main.py --video video.mp4   Usar video específico
  python src/main.py --webcam            Usar webcam
  python src/main.py --ecosystem-only    Solo ecosistema (sin video)
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
        help="Usar webcam en lugar de video"
    )
    
    parser.add_argument(
        "--ecosystem-only", "-e",
        action="store_true",
        help="Ejecutar solo el ecosistema sin video"
    )
    
    args = parser.parse_args()
    
    # Crear y ejecutar aplicación
    app = EcoVisionApp(
        video_source=args.video,
        use_webcam=args.webcam,
        ecosystem_only=args.ecosystem_only
    )
    
    app.run()


if __name__ == "__main__":
    main()
