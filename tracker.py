"""
tracker.py — Parte (b): Tracking 2D con YOLO + ByteTrack/BoT-SORT
===================================================================
Implementa tracking 2D de objetos en video o cámara web usando YOLO
para detección y los algoritmos ByteTrack o BoT-SORT (integrados en
ultralytics) para el seguimiento multi-objeto.

Funciones principales:
  - Detectar objetos con YOLO (clases configurables: personas, autos, etc.)
  - Asignar y mantener IDs únicos por objeto (tracking)
  - Dibujar una línea virtual y contar cuántos objetos la cruzan
  - Almacenar trayectorias: ID, frame, centro (X, Y) en un CSV
  - Mostrar todo en la GUI de customtkinter
"""

import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectTracker:
    """
    Clase principal de tracking 2D.

    Attributes
    ----------
    model : YOLO
        Modelo YOLO cargado (yolov8n.pt por defecto).
    classes : list[int] | None
        IDs de clases COCO a rastrear. None = todas.
    tracker_type : str
        'bytetrack' o 'botsort'.
    line_y : int
        Posición Y de la línea virtual de conteo.
    counted_ids : set
        IDs de objetos ya contados (evita doble conteo).
    cross_direction : dict
        Registro de la posición anterior de cada ID respecto a la línea.
    trajectories : defaultdict
        Diccionario {id: [(frame, x, y, timestamp), ...]} con trayectorias.
    total_count : int
        Contador total de objetos que cruzaron la línea.
    """

    def __init__(self, model_path="yolov8n.pt", classes=None,
                 tracker_type="bytetrack", conf=0.3, iou=0.5):
        """
        Parameters
        ----------
        model_path : str
            Ruta al modelo YOLO (pesos .pt).
        classes : list[int] | None
            Lista de clases COCO a filtrar. None = todas.
        tracker_type : str
            'bytetrack' o 'botsort'.
        conf : float
            Umbral de confianza para detección.
        iou : float
            Umbral de IoU para la supresión de no-máximos.
        """
        self.model = YOLO(model_path)
        self.classes = classes
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou

        # Estado del tracking
        self.line_y = 0
        self.counted_ids = set()
        self.cross_direction = {}  # id -> 'above' | 'below'
        self.trajectories = defaultdict(list)
        self.total_count = 0
        self.frame_idx = 0

    def set_counting_line(self, y):
        """Fija la línea virtual de conteo en la posición Y indicada."""
        self.line_y = y

    def reset_state(self):
        """Reinicia contadores y trayectorias para una nueva sesión."""
        self.counted_ids.clear()
        self.cross_direction.clear()
        self.trajectories.clear()
        self.total_count = 0
        self.frame_idx = 0

    def _update_counting(self, track_id, cx, cy):
        """
        Lógica de cruce de línea virtual.

        Detecta si el centro de un objeto cruza la línea horizontal
        en self.line_y. Usa el sentido (arriba->abajo o abajo->arriba)
        y evita contar el mismo ID dos veces.

        Parameters
        ----------
        track_id : int
            ID único del objeto asignado por el tracker.
        cx, cy : float
            Coordenadas del centro del bounding box.
        """
        prev = self.cross_direction.get(track_id)
        current = "above" if cy < self.line_y else "below"

        if prev is not None and prev != current:
            # Hubo cruce
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_count += 1

        self.cross_direction[track_id] = current

    def process_frame(self, frame):
        """
        Procesa un frame de video: detecta, trackea, cuenta y dibuja.

        Returns
        -------
        frame : np.ndarray
            Frame anotado con bounding boxes, IDs, trayectorias y línea.
        detections : list[dict]
            Lista de detecciones del frame actual:
            {'id', 'class_name', 'confidence', 'bbox': (x1,y1,x2,y2),
             'center': (cx, cy), 'frame': frame_idx}
        """
        self.frame_idx += 1
        h, w = frame.shape[:2]
        if self.line_y == 0:
            self.line_y = h // 2  # Por defecto, línea en el medio

        # --- Detección + Tracking ---
        results = self.model.track(
            frame, persist=True, tracker=self.tracker_type + ".yaml",
            conf=self.conf, iou=self.iou, classes=self.classes, verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                # Coordenadas del bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = self.model.names.get(cls_id, str(cls_id))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # --- Conteo por cruce de línea ---
                self._update_counting(track_id, cx, cy)

                # --- Guardar trayectoria ---
                self.trajectories[track_id].append(
                    (self.frame_idx, cx, cy, time.time())
                )

                # --- Dibujar bounding box + ID ---
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {class_name} {conf_val:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                detections.append({
                    "id": track_id,
                    "class_name": class_name,
                    "confidence": round(conf_val, 3),
                    "bbox": (x1, y1, x2, y2),
                    "center": (round(cx, 1), round(cy, 1)),
                    "frame": self.frame_idx,
                })

        # --- Dibujar línea virtual de conteo ---
        cv2.line(frame, (0, self.line_y), (w, self.line_y),
                 (0, 0, 255), 2)
        cv2.putText(frame, f"Conteo: {self.total_count}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- Dibujar trayectorias ---
        for tid, pts in self.trajectories.items():
            if len(pts) > 1:
                # Dibujar los últimos N puntos de la trayectoria
                trail = pts[-60:]  # últimos 60 frames
                for i in range(1, len(trail)):
                    _, x_prev, y_prev, _ = trail[i - 1]
                    _, x_curr, y_curr, _ = trail[i]
                    cv2.line(frame,
                             (int(x_prev), int(y_prev)),
                             (int(x_curr), int(y_curr)),
                             (255, 200, 0), 1)

        return frame, detections

    def save_trajectories_csv(self, filepath):
        """
        Guarda todas las trayectorias recolectadas en un CSV.

        Columnas: track_id, frame, center_x, center_y, timestamp
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame", "center_x", "center_y",
                             "timestamp"])
            for tid, pts in self.trajectories.items():
                for pt in pts:
                    frame_num, cx, cy, ts = pt
                    writer.writerow([tid, frame_num, round(cx, 1),
                                     round(cy, 1), round(ts, 3)])