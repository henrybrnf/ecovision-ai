# 🎯 EcoVision AI

## Sistema Inteligente de Vigilancia con Detección de Objetos y Agentes Evolutivos

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)]()

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Stack Tecnológico](#stack-tecnológico)
4. [Requisitos Previos](#requisitos-previos)
5. [Instalación y Configuración](#instalación-y-configuración)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Guía de Uso](#guía-de-uso)
8. [Módulos del Sistema](#módulos-del-sistema)
9. [Flujo de Ejecución](#flujo-de-ejecución)
10. [Configuración Avanzada](#configuración-avanzada)
11. [Pruebas](#pruebas)
12. [Contribución](#contribución)
13. [Autores](#autores)
14. [Licencia](#licencia)

---

## 📖 Descripción del Proyecto

**EcoVision AI** es un sistema inteligente de vigilancia que integra múltiples técnicas de Inteligencia Artificial para crear un ecosistema de monitoreo adaptativo. El sistema combina:

- **Detección de objetos en tiempo real** mediante Redes Neuronales Convolucionales (YOLO)
- **Evaluación de situaciones** utilizando Lógica Difusa
- **Agentes virtuales evolutivos** que aprenden a patrullar mediante Algoritmos Genéticos
- **Cerebros neuronales** para la toma de decisiones de cada agente

### Objetivo Principal

Desarrollar un prototipo funcional que demuestre la integración efectiva de cuatro paradigmas de Inteligencia Artificial trabajando de manera coordinada para resolver un problema de vigilancia inteligente.

### 📚 Documentación Clave (Lectura Obligatoria)
| Documento | Contenido Principal |
|-----------|---------------------|
| [**FUNDAMENTOS_ACADEMICOS.md**](FUNDAMENTOS_ACADEMICOS.md) | **MATRIZ DE TÉCNICAS (Sección 4)**, Marco Teórico y Matemáticas. |
| [**IMPACTO_Y_APLICACIONES.md**](IMPACTO_Y_APLICACIONES.md) | **JUSTIFICACIÓN TÉCNICA**, Casos de Uso (Smart Cities, Robótica) y Ética. |
| [**README.md**](README.md) | Guía de instalación y uso técnico. |

### Características Principales

| Característica | Descripción |
|----------------|-------------|
| 🎥 Detección en Tiempo Real | Procesamiento de video con YOLOv8 |
| 🌀 Sistema de Alertas Difuso | Evaluación de riesgo con lógica difusa |
| 🤖 Agentes Evolutivos | Entidades que evolucionan para mejorar |
| 📊 Dashboard Visual | Interfaz de monitoreo en tiempo real |
| 🧬 Neuroevolución | Optimización de comportamientos mediante AG |

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ECOVISION AI - ARQUITECTURA                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   ENTRADA    │    │   DETECTOR   │    │    FUZZY     │    │  SALIDA   │ │
│  │              │───▶│     CNN      │───▶│    LOGIC     │───▶│           │ │
│  │  Video/Cam   │    │   (YOLO)     │    │   Sistema    │    │  Alertas  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                              │                   │                          │
│                              ▼                   ▼                          │
│                      ┌───────────────────────────────────┐                  │
│                      │      ECOSISTEMA EVOLUTIVO         │                  │
│                      │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │                  │
│                      │  │ 🤖  │ │ 🤖  │ │ 🤖  │ │ 🤖  │  │                  │
│                      │  │Agent│ │Agent│ │Agent│ │Agent│  │                  │
│                      │  └─────┘ └─────┘ └─────┘ └─────┘  │                  │
│                      │         ▲                         │                  │
│                      │         │ Algoritmo Genético      │                  │
│                      │         │ Evolución               │                  │
│                      └───────────────────────────────────┘                  │
│                                        │                                     │
│                                        ▼                                     │
│                      ┌───────────────────────────────────┐                  │
│                      │        VISUALIZACIÓN              │                  │
│                      │   Dashboard + Simulación 2D       │                  │
│                      └───────────────────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

1. **Entrada**: Video en tiempo real o archivo de video
2. **Detección**: YOLOv8 identifica personas y objetos
3. **Evaluación**: Sistema difuso determina nivel de alerta
4. **Ecosistema**: Agentes responden a la situación
5. **Evolución**: Los mejores agentes se reproducen
6. **Visualización**: Dashboard muestra todo en tiempo real

---

## 🛠️ Stack Tecnológico

### Lenguaje y Entorno

| Componente | Versión | Descripción |
|------------|---------|-------------|
| Python | 3.10+ | Lenguaje principal |
| pip | 23.0+ | Gestor de paquetes |
| venv | Incluido | Entorno virtual |
| Git | 2.40+ | Control de versiones |

### Librerías Principales

```
# Detección de Objetos
ultralytics>=8.0.0          # YOLOv8 para detección
opencv-python>=4.8.0        # Procesamiento de video

# Lógica Difusa
scikit-fuzzy>=0.4.2         # Motor de inferencia difusa
numpy>=1.24.0               # Operaciones numéricas

# Simulación y Visualización
pygame>=2.5.0               # Simulación 2D
matplotlib>=3.7.0           # Gráficos
plotly>=5.15.0              # Gráficos interactivos

# Interfaz de Usuario
streamlit>=1.25.0           # Dashboard web (opcional)
gradio>=3.40.0              # Interfaz alternativa

# Utilidades
pandas>=2.0.0               # Manejo de datos
tqdm>=4.65.0                # Barras de progreso
pyyaml>=6.0.0               # Configuración
```

### Herramientas de Desarrollo

| Herramienta | Uso |
|-------------|-----|
| VS Code | IDE recomendado |
| Jupyter Notebook | Desarrollo interactivo |
| pytest | Testing |
| black | Formateo de código |
| flake8 | Linting |

---

## 📋 Requisitos Previos

### Hardware Recomendado

- **CPU**: Intel i5 / AMD Ryzen 5 o superior
- **RAM**: 8 GB mínimo (16 GB recomendado)
- **GPU**: NVIDIA con CUDA (opcional, acelera YOLO)
- **Almacenamiento**: 2 GB libres
- **Webcam**: Opcional (para demo en vivo)

### Software Requerido

1. **Python 3.10 o superior**
   ```bash
   python --version
   ```

2. **Git**
   ```bash
   git --version
   ```

3. **pip actualizado**
   ```bash
   python -m pip install --upgrade pip
   ```

---

## 🚀 Instalación y Configuración

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/[TU_USUARIO]/ecovision-ai.git
cd ecovision-ai
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar Modelo YOLO

```bash
# Se descarga automáticamente en la primera ejecución
# O manualmente:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Paso 5: Verificar Instalación

```bash
python -c "import cv2; import numpy; import skfuzzy; import pygame; print('✅ Todo instalado correctamente')"
```

---

## 📁 Estructura del Proyecto

```
ecovision-ai/
│
├── 📄 README.md                    # Este archivo
├── 📄 FUNDAMENTOS_ACADEMICOS.md    # Documentación académica
├── 📄 requirements.txt             # Dependencias
├── 📄 setup.py                     # Instalación como paquete
├── 📄 .gitignore                   # Archivos ignorados
├── 📄 LICENSE                      # Licencia MIT
│
├── 📂 src/                         # Código fuente principal
│   ├── 📄 __init__.py
│   ├── 📄 main.py                  # Punto de entrada
│   ├── 📄 config.py                # Configuración global
│   │
│   ├── 📂 detector/                # Módulo de Detección CNN
│   │   ├── 📄 __init__.py
│   │   ├── 📄 yolo_detector.py     # Detector YOLOv8
│   │   └── 📄 video_processor.py   # Procesamiento de video
│   │
│   ├── 📂 fuzzy_logic/             # Módulo de Lógica Difusa
│   │   ├── 📄 __init__.py
│   │   ├── 📄 fuzzy_system.py      # Sistema difuso
│   │   ├── 📄 membership.py        # Funciones de membresía
│   │   └── 📄 rules.py             # Reglas difusas
│   │
│   ├── 📂 ecosystem/               # Módulo de Vida Artificial
│   │   ├── 📄 __init__.py
│   │   ├── 📄 agent.py             # Agente con red neuronal
│   │   ├── 📄 neural_brain.py      # Cerebro del agente
│   │   ├── 📄 world.py             # Mundo virtual
│   │   ├── 📄 genetics.py          # Algoritmo genético
│   │   └── 📄 simulation.py        # Simulación principal
│   │
│   └── 📂 visualization/           # Módulo de Visualización
│       ├── 📄 __init__.py
│       ├── 📄 dashboard.py         # Dashboard principal
│       ├── 📄 renderer.py          # Renderizado 2D
│       └── 📄 charts.py            # Gráficos de evolución
│
├── 📂 data/                        # Datos
│   ├── 📂 videos/                  # Videos de prueba
│   │   └── 📄 sample.mp4
│   └── 📂 configs/                 # Configuraciones
│       └── 📄 default.yaml
│
├── 📂 models/                      # Modelos entrenados
│   ├── 📄 yolov8n.pt               # Modelo YOLO
│   └── 📂 best_agents/             # Mejores agentes guardados
│
├── 📂 notebooks/                   # Jupyter Notebooks
│   ├── 📄 01_detector_demo.ipynb
│   ├── 📄 02_fuzzy_demo.ipynb
│   ├── 📄 03_ecosystem_demo.ipynb
│   └── 📄 04_integrated_demo.ipynb
│
├── 📂 tests/                       # Pruebas unitarias
│   ├── 📄 __init__.py
│   ├── 📄 test_detector.py
│   ├── 📄 test_fuzzy.py
│   ├── 📄 test_ecosystem.py
│   └── 📄 test_integration.py
│
├── 📂 docs/                        # Documentación adicional
│   ├── 📄 api_reference.md
│   ├── 📄 architecture.md
│   └── 📂 images/                  # Imágenes para docs
│
└── 📂 outputs/                     # Salidas del sistema
    ├── 📂 logs/                    # Logs de ejecución
    ├── 📂 screenshots/             # Capturas
    └── 📂 evolution_data/          # Datos de evolución
```

---

## 📘 Guía de Uso

### Ejecución Básica

```bash
# Activar entorno virtual
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Ejecutar aplicación principal
python src/main.py
```

### Modos de Ejecución

#### 1. Demo con Video de Prueba
```bash
python src/main.py --mode demo --video data/videos/sample.mp4
```

#### 2. Webcam en Tiempo Real
```bash
python src/main.py --mode webcam
```

#### 3. Solo Ecosistema (sin detección)
```bash
python src/main.py --mode ecosystem-only
```

#### 4. Solo Detector (sin ecosistema)
```bash
python src/main.py --mode detector-only --video data/videos/sample.mp4
```

### Controles de la Interfaz

| Tecla | Acción |
|-------|--------|
| `SPACE` | Pausar/Reanudar simulación |
| `R` | Reiniciar ecosistema |
| `S` | Guardar mejores agentes |
| `+/-` | Aumentar/Disminuir velocidad |
| `ESC` | Salir |

---

## 🧩 Módulos del Sistema

### 1. Módulo Detector (CNN - YOLO)

```python
from src.detector import YOLODetector

detector = YOLODetector(model_path='yolov8n.pt')
detections = detector.detect(frame)
# Retorna: lista de objetos detectados con posición y confianza
```

**Funcionalidades:**
- Detección de personas, vehículos, objetos
- Tracking entre frames
- Extracción de métricas (conteo, posiciones, velocidades)

### 2. Módulo Fuzzy Logic

```python
from src.fuzzy_logic import AlertSystem

alert_system = AlertSystem()
alert_level = alert_system.evaluate(
    person_count=5,
    movement_speed=0.7,
    zone_risk=0.4
)
# Retorna: nivel de alerta (0.0 a 1.0)
```

**Variables Lingüísticas:**
- `person_count`: bajo, medio, alto
- `movement_speed`: lento, moderado, rápido
- `zone_risk`: segura, neutral, peligrosa
- `alert_level`: normal, precaución, alerta, emergencia

### 3. Módulo Ecosistema Evolutivo

```python
from src.ecosystem import Simulation

sim = Simulation(
    world_size=(800, 600),
    agent_count=20,
    generations=100
)
sim.run()
```

**Componentes:**
- `Agent`: Entidad con cerebro neural
- `NeuralBrain`: Red neuronal feedforward
- `GeneticAlgorithm`: Evolución de agentes
- `World`: Ambiente de simulación

### 4. Módulo Visualización

```python
from src.visualization import Dashboard

dashboard = Dashboard()
dashboard.update(
    frame=video_frame,
    detections=detections,
    alert_level=alert_level,
    agents=ecosystem.agents,
    stats=evolution_stats
)
dashboard.render()
```

---

## 🔄 Flujo de Ejecución

```
INICIO
   │
   ▼
┌─────────────────────┐
│ 1. Cargar Video/Cam │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ 2. Inicializar      │
│    - Detector YOLO  │
│    - Sistema Difuso │
│    - Ecosistema     │
└─────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│            LOOP PRINCIPAL               │
│  ┌───────────────────────────────────┐  │
│  │ 3. Capturar Frame                 │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │ 4. Detectar Objetos (YOLO)        │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │ 5. Evaluar Situación (Fuzzy)      │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │ 6. Actualizar Ecosistema          │  │
│  │    - Mover agentes                │  │
│  │    - Evaluar fitness              │  │
│  │    - Evolucionar si corresponde   │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │ 7. Renderizar Dashboard           │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│         ¿Continuar? ──NO──▶ FIN        │
│              │                         │
│             YES                        │
│              │                         │
│              ▼                         │
│         Volver al paso 3               │
└─────────────────────────────────────────┘
```

---

## ⚙️ Configuración Avanzada

### Archivo de Configuración: `data/configs/default.yaml`

```yaml
# Configuración del Detector
detector:
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  classes: [0]  # 0 = personas
  
# Configuración del Sistema Difuso
fuzzy:
  person_count_max: 20
  movement_speed_max: 1.0
  defuzzification_method: "centroid"

# Configuración del Ecosistema
ecosystem:
  world_width: 800
  world_height: 600
  agent_count: 20
  mutation_rate: 0.1
  crossover_rate: 0.7
  elitism: 2
  
# Configuración de la Red Neuronal del Agente
agent_brain:
  input_size: 8   # sensores
  hidden_size: 16
  output_size: 4  # acciones

# Configuración de Visualización
visualization:
  fps: 30
  show_detections: true
  show_agents: true
  show_stats: true
```

---

## 🧪 Pruebas

### Ejecutar Todas las Pruebas

```bash
pytest tests/ -v
```

### Ejecutar Pruebas por Módulo

```bash
# Solo detector
pytest tests/test_detector.py -v

# Solo fuzzy
pytest tests/test_fuzzy.py -v

# Solo ecosistema
pytest tests/test_ecosystem.py -v

# Integración
pytest tests/test_integration.py -v
```

### Cobertura de Código

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

---

## 👥 Autores

| Nombre | Rol | Contacto |
|--------|-----|----------|
| Henry Nuñez | Desarrollador Principal | henrybrnf@gmail.com |

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📞 Soporte

Para reportar bugs o solicitar features, crear un [Issue](https://github.com/[TU_USUARIO]/ecovision-ai/issues).

---

**Desarrollado con ❤️ para el curso de Inteligencia Artificial**
