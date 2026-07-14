"""
speed.py — Parte (c): Cálculo de velocidad en tiempo real + mapa de calor
===========================================================================
Calcula la velocidad de cada objeto seguido en píxeles/segundo usando
las trayectorias del tracker. Integra los resultados en el frame de video
y construye un mapa de calor acumulativo (heatmap) de las zonas por donde
transitan los objetos.

Funciones principales:
  1. Velocidad instantánea por objeto (px/s) con la distancia recorrida
     entre frames consecutivos dividida por el delta de tiempo real.
  2. Conversión a m/s si se conoce el factor de escala (px/m).
  3. Mapa de calor acumulativo con cv2.addWeighted al frame.
  4. Registro de zonas visitadas por cada objeto.
"""

import time
import cv2
import numpy as np
from collections import defaultdict


class SpeedCalculator:
    """
    Calculadora de velocidad en tiempo real + generador de mapa de calor.

    Attributes
    ----------
    px_per_meter : float | None
        Factor de escala. Si es None, solo se reporta en px/s.
    speeds : dict[int, float]
        Velocidad actual (px/s o m/s) por ID de objeto.
    last_positions : dict[int, tuple]
        Última posición (x, y) de cada objeto + timestamp.
    heat_accum : np.ndarray | None
        Imagen de acumulación para el mapa de calor.
    zone_visits : defaultdict[int, list]
        Registra las zonas (buckets de píxeles) por donde pasa cada ID.
    heatmap_size : tuple
        Dimensiones del frame para redimensionar el acumulador de calor.
    """

    def __init__(self, px_per_meter=None, decay=0.985, heat_radius=15, zone_grid_size=50):
        """
        Parameters
        ----------
        px_per_meter : float | None
            Cuántos píxeles equivalen a 1 metro en la escena.
            Si es None, la velocidad se reporta solo en px/s.
        decay : float
            Factor de decaimiento del heatmap (0-1). Cerca de 1 el
            calor persiste más; valores menores lo desvanecen rápido.
        heat_radius : int
            Radio del kernel gaussiano depositado en el heatmap
            por cada posición de objeto.
        zone_grid_size : int
            Tamaño en píxeles de cada celda de la grilla de zonas.
        """
        self.px_per_meter = px_per_meter
        self.decay = decay
        self.heat_radius = heat_radius
        self.zone_grid_size = zone_grid_size

        self.speeds = {}
        self.last_positions = {}  # id -> (x, y, timestamp)
        self.heat_accum = None
        self.zone_visits = defaultdict(list)
        self.zone_freq = defaultdict(int)  # (zx, zy) -> contador de visitas
        self._gaussian_kernel = self._make_gaussian_kernel(heat_radius)

    def init_heatmap(self, height, width):
        """Inicializa (o redimensiona) el acumulador del mapa de calor."""
        if self.heat_accum is None or self.heat_accum.shape[:2] != (height, width):
            self.heat_accum = np.zeros((height, width), dtype=np.float32)

    def _make_gaussian_kernel(self, radius):
        """Crea un kernel gaussiano 2D para el depósito de calor."""
        size = 2 * radius + 1
        ax = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(ax, ax)
        sigma = radius / 2.5
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.max()
        return kernel

    def reset(self):
        """Reinicia todos los estados."""
        self.speeds.clear()
        self.last_positions.clear()
        self.zone_visits.clear()
        self.zone_freq.clear()
        self.heat_accum = None

    def set_px_per_meter(self, px_per_meter):
        """Actualiza el factor de escala píxeles→metros."""
        self.px_per_meter = px_per_meter

    def compute_speed(self, track_id, cx, cy, timestamp=None):
        """
        Calcula la velocidad instantánea de un objeto.

        Usa la distancia euclidiana entre la posición anterior y la
        actual, dividida por el delta de tiempo.

        Returns
        -------
        speed_px_s : float
            Velocidad en píxeles por segundo.
        speed_real : float | None
            Velocidad en m/s (si px_per_meter está definido), si no None.
        """
        if timestamp is None:
            timestamp = time.time()

        prev = self.last_positions.get(track_id)
        if prev is None:
            self.last_positions[track_id] = (cx, cy, timestamp)
            return 0.0, None

        px, py, pt = prev
        dt = timestamp - pt
        if dt <= 0:
            return 0.0, None

        dist_px = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        speed_px_s = dist_px / dt

        speed_real = None
        if self.px_per_meter and self.px_per_meter > 0:
            speed_real = speed_px_s / self.px_per_meter  # m/s

        self.last_positions[track_id] = (cx, cy, timestamp)
        self.speeds[track_id] = speed_px_s
        return speed_px_s, speed_real

    def update_heatmap(self, cx, cy):
        """
        Depósita un punto de calor con kernel gaussiano en (cx, cy).

        Usa un kernel gaussiano real para que los puntos se fundan
        suavemente donde se superponen, generando un gradiente natural.
        """
        if self.heat_accum is None:
            return
        ix, iy = int(cx), int(cy)
        h, w = self.heat_accum.shape[:2]

        r = self.heat_radius
        kernel = self._gaussian_kernel

        # Calcular la región de superposición entre el kernel y el acumulador
        x1 = max(0, ix - r)
        y1 = max(0, iy - r)
        x2 = min(w, ix + r + 1)
        y2 = min(h, iy + r + 1)

        kx1 = x1 - (ix - r)
        ky1 = y1 - (iy - r)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)

        self.heat_accum[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

    def apply_decay(self):
        """Aplica el decaimiento al acumulador de calor."""
        if self.heat_accum is not None:
            self.heat_accum *= self.decay

    def get_heatmap_overlay(self, frame):
        """
        Genera la superposición del mapa de calor sobre el frame.

        Normaliza el acumulador a 0-255, aplica colormap JET y lo
        fusiona con el frame original usando addWeighted.

        Returns
        -------
        frame : np.ndarray
            Frame con el heatmap superpuesto.
        """
        if self.heat_accum is None:
            return frame

        # Normalizar a 0-255
        heat_norm = cv2.normalize(
            self.heat_accum, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        # Aplicar colormap
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        # Mezclar con el frame (30% heatmap, 70% original)
        return cv2.addWeighted(frame, 0.7, heat_color, 0.3, 0)

    def get_heatmap_image(self):
        """
        Devuelve el mapa de calor puro como imagen (sin video de fondo).

        Returns
        -------
        np.ndarray | None
            Imagen del heatmap normalizado + JET, o None si no hay datos.
        """
        if self.heat_accum is None:
            return None

        max_val = self.heat_accum.max()
        if max_val <= 0:
            return None

        heat_norm = cv2.normalize(
            self.heat_accum, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

    def annotate_frame(self, frame, detections):
        """
        Procesa todas las detecciones del frame, actualiza velocidades
        y mapa de calor, y dibuja la información sobre el frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame de video actual.
        detections : list[dict]
            Lista de detecciones del frame (desde tracker.process_frame).

        Returns
        -------
        frame : np.ndarray
            Frame con velocidades y heatmap anotados.
        """
        h, w = frame.shape[:2]
        self.init_heatmap(h, w)

        for det in detections:
            tid = det["id"]
            cx, cy = det["center"]
            frame_idx = det["frame"]

            # Calcular velocidad
            speed_px, speed_real = self.compute_speed(tid, cx, cy)

            # Actualizar mapa de calor con kernel gaussiano
            self.update_heatmap(cx, cy)

            # Registrar zona visitada
            gs = self.zone_grid_size
            zone = (int(cx) // gs, int(cy) // gs)
            self.zone_visits[tid].append(zone)
            self.zone_freq[zone] += 1

            # --- Dibujar velocidad sobre el bounding box ---
            if speed_real is not None:
                speed_kmh = speed_real * 3.6
                label = f"{speed_kmh:.1f} km/h"
            else:
                label = f"{speed_px:.0f} px/s"

            x1, y1, x2, y2 = det["bbox"]
            cv2.putText(frame, label, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2)

        # Aplicar decaimiento al heatmap y superponerlo
        self.apply_decay()
        frame = self.get_heatmap_overlay(frame)

        # Dibujar grilla de zonas con frecuencia de visitas
        frame = self._draw_zone_grid(frame)

        return frame

    def _draw_zone_grid(self, frame):
        """
        Dibuja una grilla semitransparente donde cada celda se colorea
        según la frecuencia acumulada de visitas. Las celdas más
        visitadas se ven más rojas/calientes.
        """
        if not self.zone_freq:
            return frame

        gs = self.zone_grid_size
        h, w = frame.shape[:2]

        # Encontrar el máximo de visitas para normalizar
        max_freq = max(self.zone_freq.values()) if self.zone_freq else 1

        # Crear overlay para la grilla
        overlay = frame.copy()

        for (zx, zy), freq in self.zone_freq.items():
            x1 = zx * gs
            y1 = zy * gs
            x2 = min(x1 + gs, w)
            y2 = min(y1 + gs, h)

            # Intensidad normalizada
            alpha = freq / max_freq  # 0.0 a 1.0

            # Color: verde (poco) → amarillo → rojo (mucho)
            if alpha < 0.5:
                r = int(alpha * 2 * 255)
                g = 255
                b = 0
            else:
                r = 255
                g = int((1.0 - alpha) * 2 * 255)
                b = 0

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 1)

        # Mezclar overlay al 25%
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        return frame

    def get_all_speeds(self):
        """Retorna un dict {id: (speed_px_s, speed_real)} con todos los objetos."""
        result = {}
        for tid, spx in self.speeds.items():
            sr = spx / self.px_per_meter if self.px_per_meter and \
                 self.px_per_meter > 0 else None
            result[tid] = (round(spx, 1), round(sr, 2) if sr else None)
        return result

    def get_zone_report(self):
        """
        Genera un reporte de zonas visitadas por cada objeto.

        Returns
        -------
        report : dict[int, list[tuple]]
            {id: [(zone_x, zone_y), ...]} zonas únicas por objeto.
        """
        report = {}
        for tid, zones in self.zone_visits.items():
            report[tid] = list(set(zones))
        return report