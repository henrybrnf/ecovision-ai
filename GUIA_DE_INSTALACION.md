# 🛠️ Guía de Instalación y Despliegue - EcoVision AI

Esta guía detalla paso a paso cómo configurar el proyecto **EcoVision AI** en una computadora nueva (Windows, Mac o Linux) desde cero.

---

## 📋 1. Requisitos Previos

Antes de empezar, asegúrate de tener instalado lo siguiente:

### A. Python (3.10 o superior)
El cerebro del proyecto.
1.  Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/).
2.  **IMPORTANTE**: Al instalar en Windows, marca la casilla **"Add Python to PATH"** en la primera pantalla.

### B. Git
Para descargar el código.
1.  Descarga desde [git-scm.com](https://git-scm.com/downloads).
2.  Instala con las opciones por defecto.

### C. VS Code (Opcional pero recomendado)
El editor de código.
1.  Descarga desde [code.visualstudio.com](https://code.visualstudio.com/).

---

## 🚀 2. Descargar el Proyecto

Abre una terminal (PowerShell en Windows, Terminal en Mac/Linux).

### Opción A: Usando HTTPS (Fácil)
Si no tienes llaves SSH configuradas:

```bash
cd Documentos
git clone https://github.com/henrybrnf/ecovision-ai.git
cd ecovision-ai
```

### Opción B: Usando SSH (Avanzado)
Si ya configuraste tus llaves SSH en GitHub:

```bash
cd Documentos
git clone git@github.com:henrybrnf/ecovision-ai.git
cd ecovision-ai
```

---

## ⚡ 3. Configurar el Entorno Virtual

Es vital crear un "entorno virtual" para no mezclar librerías con otras cosas de tu PC.

### En Windows:

1.  **Crear el entorno:**
    ```powershell
    python -m venv .venv
    ```
    *(Nota: El punto antes de `venv` es por convención, pero opcional)*.

2.  **Activar el entorno:**
    ```powershell
    .venv\Scripts\activate
    ```
    Verás que tu terminal ahora dice `(.venv)` al principio.

### En Mac / Linux:

1.  **Crear:**
    ```bash
    python3 -m venv .venv
    ```

2.  **Activar:**
    ```bash
    source .venv/bin/activate
    ```

---

## 📦 4. Instalar Dependencias

Con el entorno `(.venv)` activo, instala las librerías necesarias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto descargará e instalará herramientas como YOLO (`ultralytics`), `pygame`, `scikit-fuzzy`, etc. Puede tardar unos minutos.

---

## 🎬 5. Ejecución del Proyecto

### Prueba Rápida (Sin video)
Para verificar que todo se instaló bien sin necesitar archivos de video:

```bash
python src/main.py --mode ecosystem-only
```
Deberías ver una ventana negra con "agentes" (puntos) moviéndose.

### Prueba Completa (Con video)
El proyecto incluye scripts, pero necesitas un video.
1.  Consigue un video MP4 de personas caminando.
2.  Guárdalo en la carpeta `data/videos/`, por ejemplo: `data/videos/prueba.mp4`.
3.  Ejecuta:

```bash
python src/main.py --video "data/videos/prueba.mp4"
```

---

## ❓ Solución de Problemas Comunes

**1. Error: "python no se reconoce como un comando..."**
*   **Solución:** No marcaste "Add Python to PATH" al instalar. Reinstala Python y marca esa casilla.

**2. Error: "ModuleNotFoundError: No module named 'ultralytics'"**
*   **Solución:** No activaste el entorno virtual. Asegúrate de ejecutar `.venv\Scripts\activate` antes de `python`.

**3. La ventana se cierra inmediatamente**
*   **Solución:** Ejecuta el programa desde la terminal (PowerShell) para ver el mensaje de error, no le des doble clic al archivo `.py`.

**4. Error de Permisos en Windows al activar venv**
*   **Solución:** Abre PowerShell como Administrador y ejecuta: `Set-ExecutionPolicy RemoteSigned`, luego intenta activar de nuevo.

---

**Autor:** Henry Nuñez ([henrybrnf@gmail.com](mailto:henrybrnf@gmail.com))
**Derechos Reservados © 2025**
