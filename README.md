# Actividad 08 — Etapa 02: Tracking y Odometría Visual

## Descripción

Implementación de tracking 2D, cálculo de velocidad en tiempo real con mapa de calor, y odometría visual para estabilización con cámara en movimiento.

## Estructura

| Archivo | Parte | Descripción |
|---------|-------|-------------|
| `tracker.py` | (b) | Tracking 2D con YOLO + ByteTrack/BoT-SORT, conteo por línea virtual, guardado de trayectorias en CSV |
| `speed.py` | (c) | Cálculo de velocidad en tiempo real (px/s o km/h) + mapa de calor acumulativo |
| `odometry.py` | (d) | Odometría visual 2D: Shi-Tomasi + Lucas-Kanade + Homografía RANSAC para estabilizar video |
| `main.py` | GUI | Interfaz con customtkinter, 3 pestañas (una por cada parte) |

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Uso

### Pestaña B — Tracking 2D
- Selecciona fuente (webcam o video)
- Elige modelo YOLO y tipo de tracker (ByteTrack o BoT-SORT)
- Inicia → detecta, trackea, cuenta objetos que cruzan la línea virtual
- Guardar trayectorias en CSV

### Pestaña C — Velocidad + Calor
- Igual que B, pero calcula velocidad instantánea por objeto
- Ingresa px/metro para conversión a km/h
- Mapa de calor superpuesto mostrando zonas de mayor tránsito
- Reporte de zonas visitadas por objeto

### Pestaña D — Cámara Móvil
- Estabiliza el video con odometría visual antes del tracking
- Muestra original vs estabilizado lado a lado
- El tracker funciona sobre el video estabilizado

## Tecnologías

- Python 3.11+
- customtkinter (GUI)
- ultralytics YOLOv8 (detección + tracking)
- OpenCV (flujo óptico, homografía, visualización)
- NumPy (cálculos numéricos)