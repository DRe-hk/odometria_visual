"""
main.py — GUI con customtkinter para la Etapa 02 de la Actividad 08
===================================================================
Programa principal que integra las partes (b), (c) y (d) en una interfaz
gráfica con pestañas (tabview) usando customtkinter.

Pestañas:
  1. "Tracking 2D"      → Parte (b): YOLO + ByteTrack/BoT-SORT + conteo
  2. "Velocidad + Calor" → Parte (c): velocidad en tiempo real + heatmap
  3. "Cámara Móvil"     → Parte (d): odometría visual + tracking estabilizado

Uso:
    python main.py

Requisitos:
    pip install customtkinter ultralytics opencv-python numpy
"""

import csv
import os
import threading
import time
from datetime import datetime

import customtkinter as ctk
import cv2

import tracker as trk
import speed as spd
import odometry as odo

# --- Configuración global ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Clases COCO relevantes para tráfico y personas
COCO_PERSON = 0
COCO_CAR = 2
COCO_BUS = 5
COCO_TRUCK = 7
DEFAULT_CLASSES = [COCO_PERSON, COCO_CAR, COCO_BUS, COCO_TRUCK]


class TrackingApp(ctk.CTk):
    """
    Aplicación principal con pestañas para cada parte de la Etapa 02.
    """

    def __init__(self):
        super().__init__()
        self.title("Actividad 08 — Etapa 02: Tracking & Odometría Visual")
        self.geometry("1000x700")

        # --- Estado de ejecución ---
        self.running = False
        self.cap = None
        self.tracker = None
        self.speed_calc = None
        self.stabilizer = None
        self.prev_bboxes = None  # Para el estabilizador (parte d)

        # --- Construir UI ---
        self._build_ui()

    def _build_ui(self):
        """Construye la interfaz de usuario con pestañas."""
        # ---- Título principal ----
        title = ctk.CTkLabel(self,
                             text="🐾 Actividad 08 — Etapa 02: Tracking y Odometría Visual",
                             font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=10)

        # ---- Tabview con 3 pestañas ----
        self.tabview = ctk.CTkTabview(self, width=960, height=600)
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)

        self.tab_b = self.tabview.add("B) Tracking 2D")
        self.tab_c = self.tabview.add("C) Velocidad + Calor")
        self.tab_d = self.tabview.add("D) Cámara Móvil")

        self._build_tab_b()
        self._build_tab_c()
        self._build_tab_d()

    # ======================================================================
    # Pestaña B — Tracking 2D
    # ======================================================================
    def _build_tab_b(self):
        tab = self.tab_b

        # --- Frame de controles ---
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(controls, text="Fuente:").grid(row=0, column=0, padx=5, pady=5)
        self.source_var = ctk.StringVar(value="webcam")
        self.source_menu = ctk.CTkOptionMenu(
            controls, variable=self.source_var,
            values=["webcam", "video"]
        )
        self.source_menu.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Ruta video:").grid(row=0, column=2, padx=5, pady=5)
        self.video_path_entry = ctk.CTkEntry(
            controls, placeholder_text="ruta/al/video.mp4",
            width=200
        )
        self.video_path_entry.grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Tracker:").grid(row=0, column=4, padx=5, pady=5)
        self.tracker_type_var = ctk.StringVar(value="bytetrack")
        self.tracker_menu = ctk.CTkOptionMenu(
            controls, variable=self.tracker_type_var,
            values=["bytetrack", "botsort"]
        )
        self.tracker_menu.grid(row=0, column=5, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Modelo YOLO:").grid(row=1, column=0, padx=5, pady=5)
        self.model_var = ctk.StringVar(value="yolov8n.pt")
        self.model_menu = ctk.CTkOptionMenu(
            controls, variable=self.model_var,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
        )
        self.model_menu.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Línea Y (px):").grid(row=1, column=2, padx=5, pady=5)
        self.line_y_entry = ctk.CTkEntry(
            controls, placeholder_text="0 = auto (medio)",
            width=120
        )
        self.line_y_entry.grid(row=1, column=3, padx=5, pady=5)

        # --- Botones ---
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_start_b = ctk.CTkButton(
            btn_frame, text="▶ Iniciar Tracking",
            command=self.start_b
        )
        self.btn_start_b.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop_b = ctk.CTkButton(
            btn_frame, text="⏹ Detener",
            command=self.stop,
            fg_color="#cc3333"
        )
        self.btn_stop_b.grid(row=0, column=1, padx=5, pady=5)

        self.btn_save_b = ctk.CTkButton(
            btn_frame, text="💾 Guardar Trayectorias (CSV)",
            command=self.save_trajectories
        )
        self.btn_save_b.grid(row=0, column=2, padx=5, pady=5)

        # --- Frame del video ---
        self.video_frame_b = ctk.CTkLabel(tab, text="")
        self.video_frame_b.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Stats ---
        self.stats_label_b = ctk.CTkLabel(
            tab, text="Objetos contados: 0 | Objetos activos: 0",
            font=ctk.CTkFont(size=14)
        )
        self.stats_label_b.pack(pady=5)

    # ======================================================================
    # Pestaña C — Velocidad + Mapa de Calor
    # ======================================================================
    def _build_tab_c(self):
        tab = self.tab_c

        # --- Frame de controles ---
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(controls, text="Fuente:").grid(row=0, column=0, padx=5, pady=5)
        self.source_var_c = ctk.StringVar(value="webcam")
        self.source_menu_c = ctk.CTkOptionMenu(
            controls, variable=self.source_var_c,
            values=["webcam", "video"]
        )
        self.source_menu_c.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Ruta video:").grid(row=0, column=2, padx=5, pady=5)
        self.video_path_entry_c = ctk.CTkEntry(
            controls, placeholder_text="ruta/al/video.mp4",
            width=200
        )
        self.video_path_entry_c.grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(controls, text="px por metro:").grid(row=0, column=4, padx=5, pady=5)
        self.px_per_meter_entry = ctk.CTkEntry(
            controls, placeholder_text="ej: 50 (0=desactivado)",
            width=100
        )
        self.px_per_meter_entry.grid(row=0, column=5, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Decay calor:").grid(row=1, column=0, padx=5, pady=5)
        self.decay_slider = ctk.CTkSlider(
            controls, from_=0.9, to=0.999,
            value=0.985
        )
        self.decay_slider.grid(row=1, column=1, padx=5, pady=5)

        self.show_heatmap_var = ctk.CTkCheckBox(
            controls, text="Mostrar mapa de calor"
        )
        self.show_heatmap_var.select()
        self.show_heatmap_var.grid(row=1, column=2, padx=5, pady=5)

        # --- Botones ---
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_start_c = ctk.CTkButton(
            btn_frame, text="▶ Iniciar Velocidad + Calor",
            command=self.start_c
        )
        self.btn_start_c.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop_c = ctk.CTkButton(
            btn_frame, text="⏹ Detener",
            command=self.stop,
            fg_color="#cc3333"
        )
        self.btn_stop_c.grid(row=0, column=1, padx=5, pady=5)

        self.btn_report_c = ctk.CTkButton(
            btn_frame, text="📊 Reporte de Zonas",
            command=self.show_zone_report
        )
        self.btn_report_c.grid(row=0, column=2, padx=5, pady=5)

        # --- Frame del video ---
        self.video_frame_c = ctk.CTkLabel(tab, text="")
        self.video_frame_c.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Stats ---
        self.stats_label_c = ctk.CTkLabel(
            tab, text="Velocidades: (sin datos)",
            font=ctk.CTkFont(size=14)
        )
        self.stats_label_c.pack(pady=5)

    # ======================================================================
    # Pestaña D — Cámara Móvil (Odometría Visual)
    # ======================================================================
    def _build_tab_d(self):
        tab = self.tab_d

        # --- Frame de controles ---
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(controls, text="Fuente:").grid(row=0, column=0, padx=5, pady=5)
        self.source_var_d = ctk.StringVar(value="webcam")
        self.source_menu_d = ctk.CTkOptionMenu(
            controls, variable=self.source_var_d,
            values=["webcam", "video"]
        )
        self.source_menu_d.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Ruta video:").grid(row=0, column=2, padx=5, pady=5)
        self.video_path_entry_d = ctk.CTkEntry(
            controls, placeholder_text="ruta/al/video.mp4",
            width=200
        )
        self.video_path_entry_d.grid(row=0, column=3, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Mostrar features").grid(row=0, column=4, padx=5, pady=5)
        self.show_features_var = ctk.CTkCheckBox(
            controls, text="En/morado"
        )
        self.show_features_var.select()
        self.show_features_var.grid(row=0, column=5, padx=5, pady=5)

        ctk.CTkLabel(controls, text="Comparar orig/estab").grid(row=1, column=0, padx=5, pady=5)
        self.show_compare_var = ctk.CTkCheckBox(
            controls, text="Lado a lado"
        )
        self.show_compare_var.select()
        self.show_compare_var.grid(row=1, column=1, padx=5, pady=5)

        # --- Botones ---
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.pack(fill="x", padx=10, pady=5)

        self.btn_start_d = ctk.CTkButton(
            btn_frame, text="▶ Iniciar Cámara Móvil",
            command=self.start_d
        )
        self.btn_start_d.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop_d = ctk.CTkButton(
            btn_frame, text="⏹ Detener",
            command=self.stop,
            fg_color="#cc3333"
        )
        self.btn_stop_d.grid(row=0, column=1, padx=5, pady=5)

        self.btn_save_d = ctk.CTkButton(
            btn_frame, text="💾 Guardar Trayectorias (CSV)",
            command=self.save_trajectories
        )
        self.btn_save_d.grid(row=0, column=2, padx=5, pady=5)

        # --- Frame del video ---
        self.video_frame_d = ctk.CTkLabel(tab, text="")
        self.video_frame_d.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Stats ---
        self.stats_label_d = ctk.CTkLabel(
            tab, text="Odometría: H = I (sin movimiento)",
            font=ctk.CTkFont(size=14)
        )
        self.stats_label_d.pack(pady=5)

    # ======================================================================
    # Lógica de los botones
    # ======================================================================

    def _get_source(self, source_var, path_entry):
        """Devuelve el índice de cámara o la ruta del video según la selección."""
        if source_var.get() == "webcam":
            return 0
        path = path_entry.get().strip()
        return path if path else 0

    def _open_capture(self, source):
        """Abre el VideoCapture."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir la fuente: {source}")
        return cap

    def _frame_to_tk(self, frame, max_width=900):
        """Convierte un frame OpenCV a CTkImage para mostrarlo en un CTkLabel."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img = Image.fromarray(frame_rgb)
        # Escalar si es muy grande
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize(
                (max_width, int(img.height * ratio)),
                Image.LANCZOS
            )
        ctk_img = ctk.CTkImage(
            light_image=img, dark_image=img,
            size=img.size
        )
        return ctk_img

    def _update_video_label(self, ctk_label, frame):
        """Actualiza un CTkLabel con un frame de video."""
        ctk_img = self._frame_to_tk(frame)
        ctk_label.configure(image=ctk_img)
        ctk_label.image = ctk_img

    def _update_stats_label(self, label, text):
        """Actualiza el texto de un label de estadísticas (thread-safe)."""
        label.configure(text=text)

    # --- Inicio de cada pestaña ---

    def start_b(self):
        """Inicia el tracking 2D (parte b)."""
        if self.running:
            return
        source = self._get_source(self.source_var, self.video_path_entry)
        model_path = self.model_var.get()
        tracker_type = self.tracker_type_var.get()
        self.thread = threading.Thread(
            target=self._run_b, args=(source, model_path, tracker_type),
            daemon=True
        )
        self.thread.start()

    def start_c(self):
        """Inicia velocidad + mapa de calor (parte c)."""
        if self.running:
            return
        source = self._get_source(self.source_var_c, self.video_path_entry_c)
        model_path = self.model_var.get()
        tracker_type = self.tracker_type_var.get()
        px_per_meter = 0
        try:
            px_per_meter = float(self.px_per_meter_entry.get() or 0)
        except ValueError:
            px_per_meter = 0
        decay_val = self.decay_slider.get()

        self.thread = threading.Thread(
            target=self._run_c,
            args=(source, model_path, tracker_type, px_per_meter or None, decay_val),
            daemon=True
        )
        self.thread.start()

    def start_d(self):
        """Inicia odometría visual + tracking estabilizado (parte d)."""
        if self.running:
            return
        source = self._get_source(self.source_var_d, self.video_path_entry_d)
        model_path = self.model_var.get()
        tracker_type = self.tracker_type_var.get()

        self.thread = threading.Thread(
            target=self._run_d, args=(source, model_path, tracker_type),
            daemon=True
        )
        self.thread.start()

    # --- Stop ---

    def stop(self):
        """Detiene la ejecución del video en cualquier pestaña."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def save_trajectories(self):
        """Guarda las trayectorias en CSV."""
        if self.tracker is None:
            return
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join(
            "output",
            f"trayectorias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.tracker.save_trajectories_csv(filepath)

    def show_zone_report(self):
        """Muestra el reporte de zonas visitadas en una ventana emergente."""
        if self.speed_calc is None:
            return
        report = self.speed_calc.get_zone_report()
        win = ctk.CTkToplevel(self)
        win.title("Reporte de Zonas Visitadas")
        win.geometry("400x500")
        textbox = ctk.CTkTextbox(win, width=380, height=460)
        textbox.pack(padx=10, pady=10)
        for tid, zones in report.items():
            textbox.insert("end",
                           f"ID {tid}: {len(zones)} zonas\n")
            for z in zones[:10]:
                textbox.insert("end", f"  zona={z}\n")
            if len(zones) > 10:
                textbox.insert("end",
                               f"  ... y {len(zones)-10} más\n")
            textbox.insert("end", "\n")
        textbox.configure(state="disabled")

    # ======================================================================
    # Loops de video (corren en threads separados)
    # ======================================================================

    def _run_b(self, source, model_path, tracker_type):
        """Loop de video para la parte (b): Tracking 2D básico."""
        self.running = True
        self.cap = self._open_capture(source)
        self.tracker = trk.ObjectTracker(
            model_path=model_path,
            classes=DEFAULT_CLASSES,
            tracker_type=tracker_type,
        )

        # Configurar línea virtual
        line_y_val = self.line_y_entry.get().strip()
        if line_y_val and line_y_val != "0":
            self.tracker.set_counting_line(int(line_y_val))

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, detections = self.tracker.process_frame(frame)

            # Actualizar UI desde el hilo principal
            self._update_video_label(self.video_frame_b, frame)
            self._update_stats_label(
                self.stats_label_b,
                f"Objetos contados: {self.tracker.total_count} | "
                f"Objetos activos: {len(detections)} | "
                f"Frame: {self.tracker.frame_idx}"
            )

        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _run_c(self, source, model_path, tracker_type, px_per_meter, decay_val):
        """Loop de video para la parte (c): Velocidad + Heatmap."""
        self.running = True
        self.cap = self._open_capture(source)
        self.tracker = trk.ObjectTracker(
            model_path=model_path,
            classes=DEFAULT_CLASSES,
            tracker_type=tracker_type,
        )
        self.speed_calc = spd.SpeedCalculator(
            px_per_meter=px_per_meter,
            decay=decay_val,
        )

        show_heat = self.show_heatmap_var.get()

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Tracking
            frame, detections = self.tracker.process_frame(frame)

            # 2. Velocidad + Heatmap
            frame = self.speed_calc.annotate_frame(frame, detections)

            self._update_video_label(self.video_frame_c, frame)

            # --- Stats de velocidad ---
            speeds = self.speed_calc.get_all_speeds()
            if speeds:
                parts = []
                for tid, (spx, sreal) in speeds.items():
                    if sreal is not None:
                        parts.append(f"ID{tid}: {sreal*3.6:.1f}km/h")
                    else:
                        parts.append(f"ID{tid}: {spx:.0f}px/s")
                speed_text = " | ".join(parts[:8])
                self._update_stats_label(
                    self.stats_label_c,
                    f"Velocidades: {speed_text}"
                )
            else:
                self._update_stats_label(
                    self.stats_label_c, "Velocidades: (sin datos)"
                )

        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _run_d(self, source, model_path, tracker_type):
        """
        Loop de video para la parte (d): Odometría visual + tracking.

        Pipeline por frame:
            1. Capturar frame original.
            2. Estabilizar el frame con odometría visual (compensar movimiento
               de cámara con homografía).
            3. Aplicar tracking 2D sobre el frame ESTABILIZADO.
            4. Extraer bounding boxes para usarse como máscara en el siguiente
               frame (evitar que las regiones con objetos contaminen la
               estimación de la homografía).
            5. Mostrar original vs estabilizado lado a lado.
        """
        self.running = True
        self.cap = self._open_capture(source)
        self.tracker = trk.ObjectTracker(
            model_path=model_path,
            classes=DEFAULT_CLASSES,
            tracker_type=tracker_type,
        )
        self.stabilizer = odo.VisualOdometryStabilizer()

        show_feats = self.show_features_var.get()
        show_compare = self.show_compare_var.get()

        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Estabilizar con odometría visual
            strok_lbl, detections_bboxes = None, None
            stabilized, H = self.stabilizer.stabilize(
                frame, prev_bboxes=self.prev_bboxes
            )

            # 2. Tracking sobre el frame estabilizado
            stab_processed, detections = self.tracker.process_frame(stabilized)

            # 3. Extraer bboxes para el siguiente frame
            self.prev_bboxes = [d["bbox"] for d in detections]

            # 4. Mostrar información de la homografía
            if H is not None:
                # La translación de la homografía nos dice cuánto se movió la cam.
                tx = H[0, 2]
                ty = H[1, 2]
                h_text = f"H transl=({tx:.1f},{ty:.1f}) | Objs: {len(detections)}"
            else:
                h_text = f"Odometría: H = I (sin movimiento) | Objs: {len(detections)}"
            self._update_stats_label(self.stats_label_d, h_text)

            # 5. Visualización: comparación lado a lado o solo estabilizado
            if show_compare:
                # Redimensionar ambos al 50% de ancho para caber
                h, w = frame.shape[:2]
                small_w = w // 2
                orig_resized = cv2.resize(frame, (small_w, h))
                stab_resized = cv2.resize(stab_processed, (small_w, h))
                # Etiquetas
                cv2.putText(orig_resized, "ORIGINAL", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(stab_resized, "ESTABILIZADO", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                display = np.hstack([orig_resized, stab_resized])
            else:
                display = stab_processed

            # Opcional: dibujar puntos features detectados
            if show_feats and self.stabilizer.prev_keypoints is not None:
                for kp in self.stabilizer.prev_keypoints:
                    x, y = int(kp[0][0]), int(kp[0][1])
                    cv2.circle(display, (x, y), 3, (255, 0, 255), -1)

            self._update_video_label(self.video_frame_d, display)

        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# --- Entry point ---
if __name__ == "__main__":
    app = TrackingApp()
    app.mainloop()