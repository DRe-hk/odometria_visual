"""
odometry.py — Parte (d): Odometría visual para tracking con cámara en movimiento
=================================================================================
Cuando la cámara se mueve, el tracker del item (b) falla porque el fondo
se desplaza y el tracker confunde el movimiento de la cámara con el
movimiento de los objetos. La solución es **estabilizar el video** usando
odometría visual antes de aplicar el tracking.

Estrategia:
  1. Detectar puntos característicos (features) con el método good-features-
     to-track de Shi-Tomasi en el frame anterior.
  2. Calcular el flujo óptico (Lucas-Kanade sparse) entre el frame anterior
     y el actual para seguir esos puntos.
  3. Estimar una homografía o matriz afín con RANSAC que describa el
     movimiento global de la cámara (la transformación que alinea ambos
     frames).
  4. Aplicar la transformación inversa al frame actual para compensar el
     movimiento de la cámara — estabilizando el video — y luego alimentar
     el frame estabilizado al tracker de objetos.

Este enfoque es una forma simplificada de odometría visual 2D:
  - Las features del fondo остаются estacionarias en la escena real.
  - El flujo óptico nos dice cuánto se movieron en la imagen.
  - La homografía modela ese movimiento global → compensamos → el fondo
    queda "quieto" y el tracker solo necesita detectar el movimiento real
    de los objetos.

Notas:
  - Se usa una máscara para excluir las regiones donde hay objetos en
    movimiento (donde el flujo óptico no corresponde al fondo).
  - El acumulador de estabilización suaviza pequeños jitter.
"""

import cv2
import numpy as np


class VisualOdometryStabilizer:
    """
    Estabilizador de video basado en odometría visual 2D.

    Usa Shi-Tomasi + Lucas-Kanade + homografía RANSAC para compensar
    el movimiento de la cámara y estabilizar el video antes del tracking.

    Attributes
    ----------
    prev_gray : np.ndarray | None
        Frame anterior en escala de grises.
    prev keypoints : np.ndarray | None
        Puntos característicos detectados en el frame anterior.
    transform_accum : np.ndarray
        Transformación acumulada (matriz 3x3) para estabilizar.
    feature_params : dict
        Parámetros para goodFeaturesToTrack.
    lk_params : dict
        Parámetros para calcOpticalFlowPyrLK.
    mask_box : tuple | None
        Bounding box (x1,y1,x2,y2) del área donde hay objetos en
        movimiento, para excluir esa zona del cálculo de la homografía.
    """

    def __init__(self, max_corners=200, quality_level=0.01,
                 min_distance=30, block_size=3,
                 canvas_scale=1.15, recenter_threshold=0.25,
                 rotation_threshold_deg=15):
        """
        Parameters
        ----------
        max_corners : int
            Número máximo de features a detectar.
        quality_level : float
            Umbral de calidad mínimo para aceptar una feature.
        min_distance : int
            Distancia mínima entre features detectadas.
        block_size : int
            Tamaño de bloque para el detector Shi-Tomasi.
        canvas_scale : float
            Factor de escala del canvas de salida (>1 = más grande).
        recenter_threshold : float
            Fracción del tamaño del frame para recentrar (0.25 = 25%).
        rotation_threshold_deg : float
            Ángulo máximo acumulado en grados antes de recentrar.
        """
        self.prev_gray = None
        self.prev_keypoints = None
        self.transform_accum = np.eye(3, dtype=np.float32)
        self.canvas_scale = canvas_scale
        self.recenter_threshold = recenter_threshold
        self.rotation_threshold_deg = rotation_threshold_deg
        self.total_rotation_deg = 0.0
        self.frame_count = 0
        self.recentered = False

        # Parameters para Shi-Tomasi
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        # Parameters para Lucas-Kanade
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Máscara para excluir objetos en movimiento
        self.mask = None
        self.detection_mask = None

    def set_detection_mask(self, bboxes):
        """
        Define regiones con objetos detectados para excluirlas del
        cálculo de la homografía.

        Parameters
        ----------
        bboxes : list[tuple] | None
            Lista de (x1, y1, x2, y2) de los bounding boxes del frame
            actual, donde hay objetos en movimiento. None = sin máscara.
        """
        if bboxes is None or len(bboxes) == 0:
            self.detection_mask = None
            return
        # Se crea en el primer frame procesado
        # Se guarda como init para el primer uso
        self._pending_bboxes = bboxes

    def reset(self):
        """Reinicia el estado del estabilizador."""
        self.prev_gray = None
        self.prev_keypoints = None
        self.transform_accum = np.eye(3, dtype=np.float32)
        self.mask = None
        self.detection_mask = None
        self.total_rotation_deg = 0.0
        self.frame_count = 0

    def _build_mask(self, shape, bboxes):
        """
        Crea una máscara binaria: 255 = fondo (usar), 0 = objeto (ignorar).
        """
        mask = np.ones(shape, dtype=np.uint8) * 255
        for (x1, y1, x2, y2) in bboxes:
            # Expandir un poco el box para cubrir bordes
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(shape[1], x2 + 10)
            y2 = min(shape[0], y2 + 10)
            mask[y1:y2, x1:x2] = 0
        return mask

    def _canvas_size(self, w, h):
        """Devuelve el tamaño del canvas escalado."""
        return int(w * self.canvas_scale), int(h * self.canvas_scale)

    def _recenter(self, w, h):
        """
        Recalcula el offset de centrado para que el frame vuelva visualmente
        al centro del canvas y reinicia la transformación acumulada.
        """
        cw, ch = self._canvas_size(w, h)
        full_transform = self._center_offset @ self.transform_accum

        # Dónde está el centro del frame original en el canvas ahora
        frame_center_h = np.array([w / 2, h / 2, 1.0])
        canvas_center = full_transform @ frame_center_h

        # Cuánto hay que trasladar para llevarlo al centro del canvas
        delta_x = cw / 2 - canvas_center[0]
        delta_y = ch / 2 - canvas_center[1]

        recenter_trans = np.array([
            [1, 0, delta_x],
            [0, 1, delta_y],
            [0, 0, 1],
        ], dtype=np.float32)

        self._center_offset = recenter_trans @ full_transform
        self.transform_accum = np.eye(3, dtype=np.float32)
        self.total_rotation_deg = 0.0
        self.frame_count = 0
        self.prev_gray = None
        self.prev_keypoints = None
        self.recentered = True

    def _needs_recenter(self, w, h):
        """Verifica si la deriva acumulada excede los umbrales."""
        if abs(self.total_rotation_deg) > self.rotation_threshold_deg:
            return True
        tx = self.transform_accum[0, 2]
        ty = self.transform_accum[1, 2]
        limit_x = w * self.recenter_threshold
        limit_y = h * self.recenter_threshold
        if abs(tx) > limit_x or abs(ty) > limit_y:
            return True
        return False

    def stabilize(self, frame, prev_bboxes=None):
        """
        Procesa el frame actual y devuelve una versión estabilizada
        sobre un canvas ampliado, con recentrado periódico.
        """
        h, w = frame.shape[:2]
        cw, ch = self._canvas_size(w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Crear máscara de objetos ---
        if prev_bboxes:
            self.mask = self._build_mask(gray.shape, prev_bboxes)
        else:
            self.mask = None

        # --- Inicializar offset de centrado en el primer frame ---
        if not hasattr(self, '_center_offset'):
            offset_x = (cw - w) // 2
            offset_y = (ch - h) // 2
            self._center_offset = np.array([
                [1, 0, offset_x],
                [0, 1, offset_y],
                [0, 0, 1],
            ], dtype=np.float32)

        # --- Primer frame ---
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=self.mask, **self.feature_params
            )
            centered = cv2.warpPerspective(frame, self._center_offset, (cw, ch))
            return centered, None

        # --- Detectar features ---
        if self.prev_keypoints is None or len(self.prev_keypoints) < 10:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=self.mask, **self.feature_params
            )
            if self.prev_keypoints is None:
                self.prev_gray = gray.copy()
                centered = cv2.warpPerspective(frame, self._center_offset, (cw, ch))
                return centered, None

        # --- Flujo óptico ---
        kp_next, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_keypoints, None,
            **self.lk_params
        )

        good_old = self.prev_keypoints[status == 1]
        good_new = kp_next[status == 1]

        if len(good_new) < 4:
            self.prev_gray = gray.copy()
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=self.mask, **self.feature_params
            )
            centered = cv2.warpPerspective(frame, self._center_offset, (cw, ch))
            return centered, None

        # --- Homografía ---
        H, mask_ransac = cv2.findHomography(
            good_new, good_old, cv2.RANSAC, 5.0
        )

        if H is None:
            self.prev_gray = gray.copy()
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=self.mask, **self.feature_params
            )
            centered = cv2.warpPerspective(frame, self._center_offset, (cw, ch))
            return centered, None

        # --- Acumular transformación y rotación ---
        self.transform_accum = self.transform_accum @ H

        # Estimar rotación acumulada desde la homografía
        # H ≈ [R|t]; atan2(H[1,0], H[0,0]) da la rotación en radianes
        angle_rad = np.arctan2(H[1, 0], H[0, 0])
        self.total_rotation_deg += np.degrees(angle_rad)
        self.frame_count += 1

        # --- Verificar si necesita recentrado ---
        if self._needs_recenter(w, h):
            self._recenter(w, h)
            # Volver a iniciar tracking de features
            self.prev_gray = gray.copy()
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=self.mask, **self.feature_params
            )

        # --- Aplicar transformación completa al canvas ampliado ---
        full_transform = self._center_offset @ self.transform_accum
        stabilized = cv2.warpPerspective(frame, full_transform, (cw, ch))

        # --- Actualizar estado ---
        self.prev_gray = gray.copy()
        self.prev_keypoints = cv2.goodFeaturesToTrack(
            self.prev_gray, mask=self.mask, **self.feature_params
        )

        return stabilized, H