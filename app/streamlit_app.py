"""
EcoVision AI - Dashboard Unificado

Dashboard principal que muestra todo el sistema funcionando:
- Video con detecciones en tiempo real
- Ecosistema de agentes evolucionando
- Indicadores de alerta
- Estadísticas de evolución
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLODetector
from src.fuzzy_logic import AlertSystem
from src.ecosystem import Simulation, SimulationConfig

# Configuración de la página - DEBE SER PRIMERO
st.set_page_config(
    page_title="🎯 EcoVision AI - Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para un diseño moderno
st.markdown("""
<style>
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fondo oscuro */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 5px 0 0 0;
    }
    
    /* Paneles */
    .panel {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 15px;
    }
    .panel-title {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        padding-bottom: 8px;
    }
    
    /* Alertas */
    .alert-indicator {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    .alert-normal { 
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    .alert-precaucion { 
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: black;
    }
    .alert-alerta { 
        background: linear-gradient(135deg, #fd7e14 0%, #dc3545 100%);
        color: white;
    }
    .alert-emergencia { 
        background: linear-gradient(135deg, #dc3545 0%, #721c24 100%);
        color: white;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Stats */
    .stat-card {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
    }
    
    /* Métricas personalizadas */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Inicializa el estado de la sesión."""
    defaults = {
        'running': False,
        'detector': None,
        'alert_system': None,
        'simulation': None,
        'video_cap': None,
        'frame_count': 0,
        'total_detections': 0,
        'detection_history': [],
        'alert_history': [],
        'fitness_history': [],
        'last_frame': None,
        'last_detections': [],
        'mapped_detections': [],  # Detecciones mapeadas al ecosistema
        'current_alert': 'normal',
        'current_alert_level': 0.0,
        'skip_frames': 2,  # Procesar cada N frames para velocidad
        'frame_skip_counter': 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_components():
    """Carga los componentes de IA."""
    if st.session_state.detector is None:
        st.session_state.detector = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.25,  # Más bajo para mejor detección
            classes=[0]
        )
    
    if st.session_state.alert_system is None:
        st.session_state.alert_system = AlertSystem(max_persons=20)
    
    if st.session_state.simulation is None:
        config = SimulationConfig(
            world_width=350,
            world_height=250,
            agent_count=10,  # Menos agentes para velocidad
            steps_per_generation=100,
            max_generations=1000
        )
        st.session_state.simulation = Simulation(config=config)
        st.session_state.simulation.start()


def process_frame(frame):
    """Procesa un frame: detecta personas y evalúa alertas."""
    # Obtener dimensiones del frame
    frame_h, frame_w = frame.shape[:2]
    
    # Detectar personas
    detections = st.session_state.detector.detect(frame)
    
    # Actualizar contadores
    st.session_state.frame_count += 1
    st.session_state.total_detections += len(detections)
    st.session_state.last_detections = detections
    
    # Mapear detecciones al espacio del ecosistema (350x250)
    eco_w, eco_h = 350, 250
    mapped_positions = []
    for det in detections:
        # Escalar posición del video al ecosistema
        x = int(det.center[0] * eco_w / frame_w)
        y = int(det.center[1] * eco_h / frame_h)
        x = max(10, min(x, eco_w - 10))
        y = max(10, min(y, eco_h - 10))
        mapped_positions.append((x, y))
    
    st.session_state.mapped_detections = mapped_positions
    
    # Calcular velocidad (simulada)
    movement_speed = min(len(detections) / 6, 1.0)
    
    # Evaluar alerta con lógica difusa
    result = st.session_state.alert_system.evaluate(
        person_count=len(detections),
        movement_speed=movement_speed,
        zone_density=len(detections) / 12
    )
    
    st.session_state.current_alert = result.alert_category
    st.session_state.current_alert_level = result.alert_level
    
    # Actualizar historiales
    st.session_state.detection_history.append(len(detections))
    st.session_state.alert_history.append(result.alert_level)
    
    # Mantener historiales cortos
    if len(st.session_state.detection_history) > 50:
        st.session_state.detection_history.pop(0)
        st.session_state.alert_history.pop(0)
    
    # Actualizar ecosistema con posiciones mapeadas (solo 1 paso para velocidad)
    st.session_state.simulation.update(mapped_positions, result.alert_level)
    
    # Dibujar detecciones en el frame
    frame_with_detections = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Color según confianza
        if det.confidence > 0.5:
            color = (0, 255, 0)  # Verde
        elif det.confidence > 0.35:
            color = (0, 255, 255)  # Amarillo
        else:
            color = (0, 165, 255)  # Naranja
        
        cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
        
        # Etiqueta
        label = f"{det.confidence:.0%}"
        cv2.putText(frame_with_detections, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame_with_detections, len(detections)


def render_header():
    """Renderiza el encabezado."""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 EcoVision AI</h1>
        <p>Sistema Inteligente de Vigilancia con Detección de Objetos y Agentes Evolutivos</p>
    </div>
    """, unsafe_allow_html=True)


def render_ecosystem_canvas(width=350, height=250):
    """Renderiza el ecosistema como imagen."""
    # Crear imagen del ecosistema
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 40)  # Fondo oscuro
    
    # Dibujar grid
    for x in range(0, width, 50):
        cv2.line(canvas, (x, 0), (x, height), (50, 50, 60), 1)
    for y in range(0, height, 50):
        cv2.line(canvas, (0, y), (width, y), (50, 50, 60), 1)

    # --- DIBUJAR COMIDA ---
    if st.session_state.simulation:
        for food_pos in st.session_state.simulation.food_items:
            fx, fy = int(food_pos[0]), int(food_pos[1])
            # Dibujar un pequeño punto verde brillante
            cv2.circle(canvas, (fx, fy), 3, (0, 255, 0), -1)
            # Efecto de brillo
            cv2.circle(canvas, (fx, fy), 6, (0, 255, 0), 1)
    
    # Dibujar detecciones como puntos de interés (usando posiciones mapeadas)
    for pos in st.session_state.mapped_detections:
        x, y = pos
        
        # Círculo rojo pulsante para marcar personas detectadas
        cv2.circle(canvas, (x, y), 15, (0, 0, 150), 2)
        cv2.circle(canvas, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
    
    # --- DIBUJAR AGENTES ---
    if st.session_state.simulation and st.session_state.simulation.agents:
        for agent in st.session_state.simulation.agents:
            ax = int(agent.position[0])
            ay = int(agent.position[1])
            
            # Limitar al canvas
            ax = max(5, min(ax, width-5))
            ay = max(5, min(ay, height-5))
            
            if not agent.alive:
                # Dibujar "X" gris para agentes muertos
                size = 4
                cv2.line(canvas, (ax-size, ay-size), (ax+size, ay+size), (100, 100, 100), 2)
                cv2.line(canvas, (ax-size, ay+size), (ax+size, ay-size), (100, 100, 100), 2)
                continue

            # --- AGENTE VIVO ---
            # Cuerpo del agente
            cv2.circle(canvas, (ax, ay), 6, agent.color, -1)
            
            # Indicador de dirección
            dx = int(np.cos(agent.angle) * 10)
            dy = int(np.sin(agent.angle) * 10)
            cv2.line(canvas, (ax, ay), (ax+dx, ay+dy), (255, 255, 255), 2)

            # Barra de Energía
            energy_pct = max(0.0, min(1.0, agent.energy / agent.config.max_energy))
            bar_color = (0, 255, 0) if energy_pct > 0.5 else (0, 0, 255) # Verde o Rojo
            
            bar_len = 12
            bar_start_x = ax - 6
            bar_end_x = int(bar_start_x + (bar_len * energy_pct))
            bar_y = ay - 10
            
            # Fondo barra
            cv2.line(canvas, (bar_start_x, bar_y), (bar_start_x + bar_len, bar_y), (50, 50, 50), 2)
            # Energía actual
            if bar_end_x > bar_start_x:
                cv2.line(canvas, (bar_start_x, bar_y), (bar_end_x, bar_y), bar_color, 2)
            
    return canvas


def main():
    """Función principal del dashboard."""
    init_session_state()
    render_header()
    
    # Sidebar para controles
    with st.sidebar:
        st.header("⚙️ Controles")
        
        # Selección de video
        st.subheader("📹 Video")
        videos_dir = Path("data/videos")
        video_files = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
        
        if video_files:
            selected_video = st.selectbox(
                "Seleccionar video:",
                video_files,
                format_func=lambda x: x.name
            )
        else:
            st.warning("No hay videos en data/videos/")
            selected_video = None
        
        st.divider()
        
        # Controles
        col1, col2 = st.columns(2)
        start_btn = col1.button("▶️ Iniciar", type="primary", use_container_width=True)
        stop_btn = col2.button("⏹️ Parar", use_container_width=True)
        
        if st.button("🔄 Reiniciar", use_container_width=True):
            st.session_state.frame_count = 0
            st.session_state.total_detections = 0
            st.session_state.detection_history = []
            st.session_state.alert_history = []
            if st.session_state.simulation:
                st.session_state.simulation.reset()
            st.rerun()
        
        st.divider()
        
        # Info del sistema
        st.subheader("ℹ️ Sistema")
        st.caption("""
        **CNN**: YOLOv8 (detección)
        **Fuzzy**: Evaluación de alertas
        **GA**: Evolución de agentes
        **NN**: Cerebros de agentes
        """)
    
    # Manejo de botones
    if start_btn and selected_video:
        load_components()
        st.session_state.video_cap = cv2.VideoCapture(str(selected_video))
        st.session_state.running = True
    
    if stop_btn:
        st.session_state.running = False
        if st.session_state.video_cap:
            st.session_state.video_cap.release()
            st.session_state.video_cap = None
    
    # ========== DASHBOARD PRINCIPAL ==========
    
    # Fila 1: Video y Ecosistema
    col_video, col_eco = st.columns([1.5, 1])
    
    with col_video:
        st.markdown('<div class="panel"><div class="panel-title">📹 Detección de Personas (YOLO)</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        detection_info = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_eco:
        st.markdown('<div class="panel"><div class="panel-title">🤖 Ecosistema Evolutivo</div>', unsafe_allow_html=True)
        eco_placeholder = st.empty()
        eco_info = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fila 2: Alerta y Estadísticas
    col_alert, col_stats = st.columns([1, 2])
    
    with col_alert:
        st.markdown('<div class="panel"><div class="panel-title">🚦 Nivel de Alerta</div>', unsafe_allow_html=True)
        alert_placeholder = st.empty()
        alert_detail = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats:
        st.markdown('<div class="panel"><div class="panel-title">📊 Estadísticas</div>', unsafe_allow_html=True)
        stats_cols = st.columns(6) # Aumentar a 6 columnas
        stat_frames = stats_cols[0].empty()
        stat_persons = stats_cols[1].empty()
        stat_alive = stats_cols[2].empty() # Nuevo
        stat_food = stats_cols[3].empty()  # Nuevo
        stat_gen = stats_cols[4].empty()
        stat_fitness = stats_cols[5].empty()
        
        chart_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== LOOP DE PROCESAMIENTO ==========
    
    if st.session_state.running and st.session_state.video_cap:
        while st.session_state.running:
            ret, frame = st.session_state.video_cap.read()
            
            if not ret:
                # Reiniciar video
                st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Procesar frame
            processed_frame, person_count = process_frame(frame)
            
            # Mostrar video
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            detection_info.caption(f"👥 Detectadas: **{person_count}** personas | Frame: {st.session_state.frame_count}")
            
            # Mostrar ecosistema
            eco_canvas = render_ecosystem_canvas()
            eco_rgb = cv2.cvtColor(eco_canvas, cv2.COLOR_BGR2RGB)
            eco_placeholder.image(eco_rgb, channels="RGB", use_container_width=True)
            
            stats = st.session_state.simulation.get_statistics()
            eco_info.caption(f"🧬 Gen: **{stats.get('generation', 0)}** | Paso: {stats.get('step', 0)}")
            
            # Mostrar alerta
            alert_cat = st.session_state.current_alert
            alert_level = st.session_state.current_alert_level
            alert_class = f"alert-{alert_cat.replace('ó', 'o')}"
            
            alert_placeholder.markdown(f"""
            <div class="alert-indicator {alert_class}">
                {alert_cat.upper()}<br>
                <span style="font-size: 2.5rem;">{alert_level:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
            
            alert_detail.caption(f"Basado en: {person_count} personas, velocidad simulada")
            
            # Actualizar estadísticas
            stat_frames.metric("🖼️ Frames", st.session_state.frame_count)
            stat_persons.metric("👥 Pers.", st.session_state.total_detections)
            
            alive_count = len([a for a in st.session_state.simulation.agents if a.alive])
            food_count = len(st.session_state.simulation.food_items)
            
            stat_alive.metric("🟢 Vivos", alive_count)
            stat_food.metric("🍎 Comida", food_count)
            
            stat_gen.metric("🧬 Gen", stats.get('generation', 0))
            stat_fitness.metric("💪 Fit", f"{stats.get('best_fitness', 0):.1f}")
            
            # Gráfico de historial
            if len(st.session_state.detection_history) > 2:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.detection_history,
                    mode='lines',
                    name='Detecciones',
                    line=dict(color='#667eea', width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=[a * 10 for a in st.session_state.alert_history],
                    mode='lines',
                    name='Alerta x10',
                    line=dict(color='#dc3545', width=2)
                ))
                fig.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)',
                    showlegend=True,
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Pequeña pausa para no saturar
            time.sleep(0.05)
    
    else:
        # Estado inicial - mostrar placeholders
        video_placeholder.info("▶️ Selecciona un video y presiona 'Iniciar' en la barra lateral")
        
        # Mostrar ecosistema estático
        if st.session_state.simulation is None:
            load_components()
        
        eco_canvas = render_ecosystem_canvas()
        eco_rgb = cv2.cvtColor(eco_canvas, cv2.COLOR_BGR2RGB)
        eco_placeholder.image(eco_rgb, channels="RGB", use_container_width=True)
        eco_info.caption("⏸️ Esperando inicio...")
        
        alert_placeholder.markdown("""
        <div class="alert-indicator alert-normal">
            NORMAL<br>
            <span style="font-size: 2.5rem;">0%</span>
        </div>
        """, unsafe_allow_html=True)
        
        stat_frames.metric("🖼️ Frames", 0)
        stat_persons.metric("👥 Pers.", 0)
        stat_alive.metric("🟢 Vivos", 0)
        stat_food.metric("🍎 Comida", 0)
        stat_gen.metric("🧬 Gen", 1)
        stat_fitness.metric("💪 Fit", "0.0")


if __name__ == "__main__":
    main()
