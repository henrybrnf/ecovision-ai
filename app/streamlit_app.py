"""
EcoVision AI - Interfaz Web con Streamlit

Una interfaz moderna y detallada para visualizar el sistema
de vigilancia inteligente con todos sus componentes.

Ejecutar con:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLODetector, VideoProcessor
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig

# Configuración de la página
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .alert-normal { background-color: rgba(0,255,100,0.2); border-left-color: #00ff64; }
    .alert-precaucion { background-color: rgba(255,255,0,0.2); border-left-color: #ffff00; }
    .alert-alerta { background-color: rgba(255,165,0,0.2); border-left-color: #ffa500; }
    .alert-emergencia { background-color: rgba(255,0,0,0.2); border-left-color: #ff0000; }
    .module-box {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Inicializa el estado de la sesión."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.detector = None
        st.session_state.alert_system = None
        st.session_state.simulation = None
        st.session_state.video_processor = None
        st.session_state.running = False
        st.session_state.frame_count = 0
        st.session_state.detection_history = []
        st.session_state.alert_history = []
        st.session_state.fitness_history = []


def initialize_components():
    """Inicializa los componentes del sistema."""
    with st.spinner("🔄 Inicializando componentes de IA..."):
        # Detector YOLO
        st.session_state.detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            classes=[0]
        )
        
        # Sistema Difuso
        st.session_state.alert_system = AlertSystem(max_persons=20)
        
        # Simulación
        config = SimulationConfig(
            world_width=400,
            world_height=300,
            agent_count=15,
            steps_per_generation=200,
            max_generations=1000
        )
        st.session_state.simulation = Simulation(config=config)
        st.session_state.simulation.start()
        
        st.session_state.initialized = True
    
    st.success("✅ Componentes inicializados correctamente!")


def render_header():
    """Renderiza el encabezado."""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 EcoVision AI</h1>
        <p>Sistema Inteligente de Vigilancia con Detección de Objetos y Agentes Evolutivos</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Renderiza la barra lateral con controles."""
    st.sidebar.title("⚙️ Controles")
    
    # Sección de Video
    st.sidebar.header("📹 Fuente de Video")
    video_option = st.sidebar.radio(
        "Seleccionar fuente:",
        ["Video de prueba", "Subir archivo", "Sin video (solo ecosistema)"]
    )
    
    video_path = None
    if video_option == "Video de prueba":
        if Path("data/videos/people-walking.mp4").exists():
            video_path = "data/videos/people-walking.mp4"
        elif Path("data/videos/sample.mp4").exists():
            video_path = "data/videos/sample.mp4"
    elif video_option == "Subir archivo":
        uploaded = st.sidebar.file_uploader("Cargar video", type=['mp4', 'avi', 'mov'])
        if uploaded:
            # Guardar temporalmente
            temp_path = Path("data/videos/uploaded.mp4")
            temp_path.write_bytes(uploaded.read())
            video_path = str(temp_path)
    
    # Parámetros del Detector
    st.sidebar.header("🔍 Detector YOLO")
    confidence = st.sidebar.slider("Confianza mínima", 0.1, 1.0, 0.5, 0.1)
    
    # Parámetros del Ecosistema
    st.sidebar.header("🧬 Algoritmo Genético")
    mutation_rate = st.sidebar.slider("Tasa de mutación", 0.01, 0.5, 0.1, 0.01)
    agent_count = st.sidebar.slider("Número de agentes", 5, 30, 15)
    
    # Controles
    st.sidebar.header("🎮 Control")
    col1, col2 = st.sidebar.columns(2)
    
    start_btn = col1.button("▶️ Iniciar", use_container_width=True)
    stop_btn = col2.button("⏹️ Detener", use_container_width=True)
    reset_btn = st.sidebar.button("🔄 Reiniciar", use_container_width=True)
    
    return {
        'video_path': video_path,
        'confidence': confidence,
        'mutation_rate': mutation_rate,
        'agent_count': agent_count,
        'start': start_btn,
        'stop': stop_btn,
        'reset': reset_btn
    }


def render_theory_section():
    """Renderiza la sección de teoría."""
    st.header("📚 Fundamentos Teóricos")
    
    tabs = st.tabs(["🧠 CNN (YOLO)", "🌀 Lógica Difusa", "🧬 Algoritmo Genético", "🔗 Integración"])
    
    with tabs[0]:
        st.markdown("""
        ### Redes Neuronales Convolucionales - YOLOv8
        
        **YOLO** (You Only Look Once) es una arquitectura de detección de objetos en tiempo real.
        
        **Cómo funciona:**
        1. La imagen se divide en una cuadrícula
        2. Cada celda predice bounding boxes y probabilidades
        3. Se aplica Non-Maximum Suppression para filtrar
        
        **En nuestro sistema:** Detecta personas en el video y extrae sus posiciones.
        """)
        
        # Diagrama CNN
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 1, 1, 1, 1],
            mode='markers+text',
            marker=dict(size=[40, 50, 60, 50, 40], color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']),
            text=['Input', 'Conv', 'Pool', 'Dense', 'Output'],
            textposition='top center'
        ))
        fig.update_layout(
            title="Arquitectura Simplificada CNN",
            showlegend=False,
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("""
        ### Lógica Difusa - Sistema de Alertas
        
        La lógica difusa maneja la **incertidumbre** mediante conjuntos difusos.
        
        **Variables de entrada:**
        - `cantidad_personas`: pocas, moderadas, muchas
        - `velocidad_movimiento`: lento, moderado, rápido
        
        **Reglas IF-THEN:**
        - IF personas=muchas AND velocidad=rápida THEN alerta=EMERGENCIA
        - IF personas=pocas AND velocidad=lenta THEN alerta=NORMAL
        """)
        
        # Gráfico de funciones de membresía
        x = np.linspace(0, 20, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0, 1 - x/5), name='Pocas', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=x, y=np.exp(-((x-10)/4)**2), name='Moderadas', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=x, y=np.maximum(0, (x-12)/8), name='Muchas', fill='tozeroy'))
        fig.update_layout(
            title="Funciones de Membresía - Cantidad de Personas",
            xaxis_title="Personas",
            yaxis_title="Grado de Membresía",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.5)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("""
        ### Algoritmo Genético - Evolución de Agentes
        
        Los agentes tienen **cerebros** (redes neuronales) que evolucionan.
        
        **Proceso evolutivo:**
        1. **Selección**: Los mejores agentes sobreviven
        2. **Cruzamiento**: Combinan sus "genes" (pesos de la red)
        3. **Mutación**: Cambios aleatorios para explorar
        4. **Nueva generación**: Población renovada
        
        **Fitness**: Mide qué tan bien un agente detecta amenazas.
        """)
        
        # Diagrama de evolución
        generations = list(range(1, 11))
        fitness = [10 + i*5 + np.random.randn()*3 for i in range(10)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=generations, y=fitness, mode='lines+markers', name='Mejor Fitness'))
        fig.update_layout(
            title="Ejemplo de Evolución del Fitness",
            xaxis_title="Generación",
            yaxis_title="Fitness",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.5)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("""
        ### Integración del Sistema
        
        ```
        VIDEO → YOLO → Detecciones → LÓGICA DIFUSA → Nivel de Alerta
                           ↓                              ↓
                    ECOSISTEMA ←────────────────────────────
                           ↓
                  ALGORITMO GENÉTICO → Agentes Evolucionados
        ```
        
        **Flujo de datos:**
        1. El video es procesado por YOLO
        2. Las detecciones alimentan al sistema difuso
        3. El nivel de alerta afecta el ecosistema
        4. Los agentes que mejor responden sobreviven
        """)


def render_demo_section(params):
    """Renderiza la sección de demostración."""
    st.header("🎮 Demostración en Vivo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Video con Detecciones")
        video_placeholder = st.empty()
        detection_count = st.empty()
    
    with col2:
        st.subheader("🤖 Ecosistema de Agentes")
        ecosystem_placeholder = st.empty()
        generation_info = st.empty()
    
    # Métricas
    st.subheader("📊 Métricas en Tiempo Real")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        persons_metric = st.empty()
    with metric_cols[1]:
        alert_metric = st.empty()
    with metric_cols[2]:
        generation_metric = st.empty()
    with metric_cols[3]:
        fitness_metric = st.empty()
    
    # Gráficos
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Historial de Detecciones")
        detection_chart = st.empty()
    
    with chart_col2:
        st.subheader("🧬 Evolución del Fitness")
        fitness_chart = st.empty()
    
    return {
        'video_placeholder': video_placeholder,
        'ecosystem_placeholder': ecosystem_placeholder,
        'detection_count': detection_count,
        'generation_info': generation_info,
        'persons_metric': persons_metric,
        'alert_metric': alert_metric,
        'generation_metric': generation_metric,
        'fitness_metric': fitness_metric,
        'detection_chart': detection_chart,
        'fitness_chart': fitness_chart
    }


def run_demo(params, placeholders):
    """Ejecuta la demostración."""
    if not st.session_state.initialized:
        st.warning("⚠️ Primero inicializa los componentes")
        return
    
    video_path = params.get('video_path')
    
    # Procesar un frame
    if video_path and Path(video_path).exists():
        if st.session_state.video_processor is None:
            st.session_state.video_processor = VideoProcessor(video_path)
        
        ret, frame = st.session_state.video_processor.read()
        if not ret:
            st.session_state.video_processor.reset()
            ret, frame = st.session_state.video_processor.read()
        
        if ret:
            # Detectar
            detections = st.session_state.detector.detect(frame)
            
            # Dibujar detecciones
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det.confidence:.0%}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Mostrar video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholders['video_placeholder'].image(frame_rgb, channels="RGB", use_container_width=True)
            placeholders['detection_count'].info(f"🔍 Personas detectadas: {len(detections)}")
            
            # Evaluar alerta
            person_count = len(detections)
            movement_speed = min(person_count / 10, 1.0)
            result = st.session_state.alert_system.evaluate(
                person_count=person_count,
                movement_speed=movement_speed
            )
            
            # Actualizar ecosistema
            det_positions = [d.center for d in detections]
            st.session_state.simulation.update(det_positions, result.alert_level)
            
            # Actualizar historial
            st.session_state.detection_history.append(person_count)
            st.session_state.alert_history.append(result.alert_level)
            
            if len(st.session_state.detection_history) > 100:
                st.session_state.detection_history.pop(0)
                st.session_state.alert_history.pop(0)
            
            # Estadísticas
            stats = st.session_state.simulation.get_statistics()
            
            # Actualizar métricas
            placeholders['persons_metric'].metric("👥 Personas", person_count)
            placeholders['alert_metric'].metric("🚨 Alerta", f"{result.alert_category.upper()}")
            placeholders['generation_metric'].metric("🧬 Generación", stats.get('generation', 0))
            placeholders['fitness_metric'].metric("💪 Mejor Fitness", f"{stats.get('best_fitness', 0):.1f}")
            
            # Crear gráfico de ecosistema
            fig_eco = go.Figure()
            for agent in st.session_state.simulation.agents:
                fig_eco.add_trace(go.Scatter(
                    x=[agent.position[0]],
                    y=[agent.position[1]],
                    mode='markers',
                    marker=dict(size=10, color=f'rgb{agent.color}'),
                    name=f'Agente {agent.id}'
                ))
            fig_eco.update_layout(
                showlegend=False,
                height=300,
                xaxis=dict(range=[0, 400], showgrid=False),
                yaxis=dict(range=[0, 300], showgrid=False),
                paper_bgcolor='rgba(20,20,30,1)',
                plot_bgcolor='rgba(30,30,50,1)',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            placeholders['ecosystem_placeholder'].plotly_chart(fig_eco, use_container_width=True)
            placeholders['generation_info'].success(f"Gen {stats.get('generation', 0)} | Paso {stats.get('step', 0)}")
            
            # Gráficos de historial
            if len(st.session_state.detection_history) > 1:
                fig_det = px.line(y=st.session_state.detection_history, title="")
                fig_det.update_layout(height=200, showlegend=False,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(30,30,50,0.5)')
                placeholders['detection_chart'].plotly_chart(fig_det, use_container_width=True)
            
            best_history = stats.get('best_history', [])
            if len(best_history) > 1:
                fig_fit = px.line(y=best_history, title="")
                fig_fit.update_layout(height=200, showlegend=False,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(30,30,50,0.5)')
                placeholders['fitness_chart'].plotly_chart(fig_fit, use_container_width=True)
    else:
        # Sin video - solo ecosistema
        st.session_state.simulation.update([], 0.3)
        stats = st.session_state.simulation.get_statistics()
        
        placeholders['video_placeholder'].info("📹 No hay video cargado")
        placeholders['persons_metric'].metric("👥 Personas", 0)
        placeholders['generation_metric'].metric("🧬 Generación", stats.get('generation', 0))
        placeholders['fitness_metric'].metric("💪 Mejor Fitness", f"{stats.get('best_fitness', 0):.1f}")


def main():
    """Función principal."""
    init_session_state()
    render_header()
    
    # Sidebar
    params = render_sidebar()
    
    # Pestañas principales
    main_tabs = st.tabs(["🎮 Demo en Vivo", "📚 Teoría", "ℹ️ Acerca de"])
    
    with main_tabs[0]:
        # Botón de inicialización
        if not st.session_state.initialized:
            if st.button("🚀 Inicializar Sistema", type="primary", use_container_width=True):
                initialize_components()
                st.rerun()
        else:
            placeholders = render_demo_section(params)
            
            if params['start']:
                st.session_state.running = True
            if params['stop']:
                st.session_state.running = False
            if params['reset']:
                st.session_state.simulation.reset()
                st.session_state.detection_history.clear()
                st.session_state.alert_history.clear()
            
            if st.session_state.running:
                run_demo(params, placeholders)
                time.sleep(0.03)  # ~30 FPS
                st.rerun()
    
    with main_tabs[1]:
        render_theory_section()
    
    with main_tabs[2]:
        st.header("ℹ️ Acerca del Proyecto")
        st.markdown("""
        ### EcoVision AI
        
        **Sistema Inteligente de Vigilancia con Detección de Objetos y Agentes Evolutivos**
        
        Este proyecto integra 4 técnicas de Inteligencia Artificial:
        
        1. 🧠 **Redes Neuronales Convolucionales** - Detección de personas con YOLOv8
        2. 🌀 **Lógica Difusa** - Evaluación de nivel de alerta
        3. 🧬 **Algoritmos Genéticos** - Evolución de agentes
        4. 🤖 **Redes Neuronales** - Cerebros de agentes
        
        ---
        
        **Desarrollado para:** Curso de Inteligencia Artificial
        
        **Tecnologías:** Python, YOLOv8, scikit-fuzzy, Pygame, Streamlit
        """)


if __name__ == "__main__":
    main()
