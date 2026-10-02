import sys
import io
import winsound

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from cortex import Cortex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import joblib

HISTORY = 1000
SENSORS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
DISPLAY_SENSORS = [s for s in SENSORS if s not in ["T7", "T8"]]
BANDS       = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]
BAND_COLORS = ["#7B61FF", "#00C2A8", "#F4A261", "#E63946", "#085C35"]

BG      = "#777778"
BG2     = "#F1F1F1"
ACCENT  = "#2A2A2A"
ACCENT2 = "#7B61FF"
TEXT    = "#000000"
TEXT_DIM= "#030303"
BORDER  = "#30363D"
RED     = "#E63946"

NIVELES = {
    0: ('SIN ESTRÉS', "#62D06B"),
    1: ('BAJO',       '#2EC4B6'),
    2: ('MEDIO',      "#CEA419"),
    3: ('ALTO',       "#FF0015")
}

#  Launcher 
class Launcher:
    def __init__(self):
        self.result = None
        self._build()

    def _build(self):
        root = tk.Tk()
        self.root = root
        root.title("EMOTIV EPOC+  ·  Cortex API")
        root.configure(bg=BG)
        root.resizable(False, False)

        w, h = 580, 500         
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        header = tk.Frame(root, bg=ACCENT, height=6)
        header.pack(fill='x')

        tk.Label(root, text="EEG MONITOR", bg=BG, fg=ACCENT,
                 font=("Courier New", 20, "bold")).pack(pady=(20, 2))
        
        tk.Frame(root, bg=BORDER, height=4).pack(fill='x', padx=40, pady=(0, 18))

        form = tk.Frame(root, bg=BG)
        form.pack(padx=40, fill='x')

        def field(parent, label, default, row):
            tk.Label(parent, text=label, bg=BG, fg=TEXT_DIM,
                     font=("Courier New", 8), anchor='w').grid(
                         row=row*2, column=0, columnspan=2,
                         sticky='w', pady=(10, 2))
            var = tk.StringVar(value=default)
            entry = tk.Entry(
                parent, textvariable=var, bg=BG2, fg=TEXT,
                insertbackground=ACCENT, relief='flat',
                font=("Courier New", 11), bd=0,
                highlightthickness=1, highlightcolor=ACCENT,
                highlightbackground=BORDER
            )
            entry.grid(row=row*2+1, column=0, columnspan=2,
                       sticky='ew', ipady=6)
            return var

        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=0)

        self.v_title = field(form, "NOMBRE DEL ARCHIVO / SESION", "Sesion_001", 0)
        self.v_desc  = field(form, "DESCRIPCION ",       "",           1)

        tk.Label(form, text="DURACION (segundos)", bg=BG, fg=TEXT_DIM,
                 font=("Courier New", 8), anchor='w').grid(
                     row=4, column=0, columnspan=2, sticky='w', pady=(10, 2))

        dur_frame = tk.Frame(form, bg=BG2,
                             highlightthickness=1, highlightbackground=BORDER)
        dur_frame.grid(row=5, column=0, columnspan=2, sticky='ew')

        self.v_dur = tk.IntVar(value=60)

        tk.Button(dur_frame, text="−", bg=BG2, fg=ACCENT,
                  font=("Courier New", 13, "bold"), bd=0, relief='flat',
                  activebackground=BG2, activeforeground=ACCENT,
                  command=lambda: self.v_dur.set(max(60, self.v_dur.get()-60))
                  ).pack(side='left', padx=8)

        tk.Label(dur_frame, textvariable=self.v_dur, bg=BG2, fg=TEXT,
                 font=("Courier New", 13, "bold"), width=6).pack(side='left')

        tk.Label(dur_frame, text="s", bg=BG2, fg=TEXT_DIM,
                 font=("Courier New", 10)).pack(side='left')

        tk.Button(dur_frame, text="+", bg=BG2, fg=ACCENT,
                  font=("Courier New", 13, "bold"), bd=0, relief='flat',
                  activebackground=BG2, activeforeground=ACCENT,
                  command=lambda: self.v_dur.set(min(3600, self.v_dur.get()+60))
                  ).pack(side='left', padx=4)

        tk.Label(form, text="CARPETA DE EXPORTACION", bg=BG, fg=TEXT_DIM,
                 font=("Courier New", 8), anchor='w').grid(
                     row=6, column=0, columnspan=2, sticky='w', pady=(10, 2))

        folder_frame = tk.Frame(form, bg=BG)
        folder_frame.grid(row=7, column=0, columnspan=2, sticky='ew')
        folder_frame.columnconfigure(0, weight=1)

        self.v_folder = tk.StringVar(value='B:\\Datos')
        folder_entry = tk.Entry(
            folder_frame, textvariable=self.v_folder, bg=BG2, fg=TEXT,
            insertbackground=ACCENT, relief='flat',
            font=("Courier New", 10), bd=0,
            highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground=BORDER
        )
        folder_entry.grid(row=0, column=0, sticky='ew', ipady=6)

        tk.Button(folder_frame, text="...", bg=BG2, fg=ACCENT,
                  font=("Courier New", 9, "bold"), bd=0, relief='flat',
                  activebackground=BORDER, activeforeground=TEXT,
                  padx=8, command=self._pick_folder
                  ).grid(row=0, column=1, padx=(4, 0), ipady=6)

        # Guardar datos ?

        self.v_guardar = tk.BooleanVar(value=True)

        chk_frame = tk.Frame(root, bg=BG)
        chk_frame.pack(pady=12)

        self.chk = tk.Checkbutton(
            chk_frame,
            text="¿Guardar sesion?",
            variable=self.v_guardar,
            bg=BG, fg=ACCENT,
            selectcolor=BG2,
            activebackground=BG,
            activeforeground=ACCENT,
            font=("Courier New", 11, "bold"),
            bd=0, relief='flat',
            cursor='hand2',
            command=self._on_toggle_guardar
        )
        self.chk.pack(side='left')

        self.lbl_modo = tk.Label(
            chk_frame,
            text="Grabar y exportar CSV",
            bg=BG, fg=TEXT_DIM,
            font=("Courier New", 9)
        )
        self.lbl_modo.pack(side='left', padx=(6, 0))

        tk.Frame(root, bg=BORDER, height=3).pack(fill='x', padx=40, pady=(0, 18))

        btn = tk.Button(
            root, text="▶  INICIAR SESION",
            bg=ACCENT, fg=BG,
            font=("Courier New", 12, "bold"),
            bd=0, relief='flat', cursor='hand2',
            activebackground=ACCENT2, activeforeground=TEXT,
            padx=20, pady=10,
            command=self._on_start
        )
        btn.pack()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.mainloop()

    def _on_toggle_guardar(self):
        if self.v_guardar.get():
            self.lbl_modo.config(text="Grabar y exportar CSV", fg="#181818")
        else:
            self.lbl_modo.config(text="Visualización", fg="#1D1D1D")

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Selecciona carpeta de exportacion")
        if folder:
            self.v_folder.set(folder.replace('/', '\\'))

    def _on_start(self):
        title = self.v_title.get().strip()
        if not title:
            messagebox.showerror("Error", "El nombre de la sesion no puede estar vacio.")
            return
        if self.v_dur.get() < 10:
            messagebox.showerror("Error", "La duracion minima es 60 segundos.")
            return
        self.result = {
            'record_title'      : title,
            'record_description': self.v_desc.get().strip(),
            'record_duration_s' : self.v_dur.get(),
            'export_folder'     : self.v_folder.get().strip(),
            'guardar_datos'     : self.v_guardar.get(),  
        }
        self.root.destroy()

    def _on_close(self):
        self.result = None
        self.root.destroy()

#  Monitor EEG 
class MonitorEEG:

    def __init__(self, client_id, client_secret,
                 record_title='Registro EEG',
                 record_description='',
                 record_duration_s=60,
                 export_folder='B:\\Datos',
                 export_data_types=None,
                 export_format='CSV',
                 export_version='V2',
                 guardar_datos=True):       

        self.client_id     = client_id
        self.client_secret = client_secret

        self.record_title       = record_title
        self.record_description = record_description
        self.record_duration_s  = record_duration_s
        self.export_folder      = export_folder
        self.export_data_types  = export_data_types or ['BP']
        self.export_format      = export_format
        self.export_version     = export_version
        self.record_id          = None
        self.guardar_datos      = guardar_datos 

        self.recording      = False
        self.elapsed        = 0
        self.session_active = False

        self.buffers = {
            sensor: {band: deque([0.0] * HISTORY, maxlen=HISTORY)
                     for band in BANDS}
            for sensor in SENSORS
        }
        self.lock = threading.Lock()

        #  Modelo 
        print('Cargando modelo XGBoost...')
        self.model  = joblib.load(r'B:\Datos\modelo_XGBoost.pkl')
        self.scaler = joblib.load(r'B:\Datos\scaler_XGBoost.pkl')
        self.feature_cols = [f'POW.{s}.{b}'
                             for s in SENSORS
                             for b in ['Theta', 'Alpha', 'BetaL', 'BetaH']]

        self.sensor_idx = {s: i for i, s in enumerate(SENSORS)}
        self.band_idx   = {b: j for j, b in enumerate(BANDS)}

        self.nivel_actual = 0
        self.nivel_texto  = 'SIN ESTRÉS'
        self.nivel_color  = '#2EC4B6'
        self.pred_lock    = threading.Lock()
        
        print(f'ML Modelo cargado — {len(self.feature_cols)} features: {self.feature_cols}')
        print('Cargando features...')

        self.model  = joblib.load(r'B:\Datos\modelo_XGBoost.pkl')
        self.scaler = joblib.load(r'B:\Datos\scaler_XGBoost.pkl')

        self.feature_cols = list(self.scaler.feature_names_in_)
        self.sensor_idx = {s: i for i, s in enumerate(SENSORS)}
        self.band_idx   = {b: j for j, b in enumerate(BANDS)}

        self.c = Cortex(self.client_id, self.client_secret, debug_mode=True)
        self.c.bind(create_session_done              = self.on_create_session_done)
        self.c.bind(create_record_done               = self.on_create_record_done)
        self.c.bind(stop_record_done                 = self.on_stop_record_done)
        self.c.bind(warn_record_post_processing_done = self.on_post_processing_done)
        self.c.bind(export_record_done               = self.on_export_record_done)
        self.c.bind(new_pow_data                     = self.on_new_pow_data)
        self.c.bind(inform_error                     = self.on_inform_error)

    #  Cortex callbacks 
    def on_create_session_done(self, *args, **kwargs):
        print('OK Sesion creada')
        self.session_active = True
        self.c.sub_request(['pow'])

        if self.guardar_datos:
            # modo GRABACION
            print(f'REC Iniciando grabacion: "{self.record_title}"')
            self.c.create_record(self.record_title, description=self.record_description)
        else:
            # modo VISUALIZACION:
            print('[VIS] Modo solo visualizacion — no se creara record')
            self.recording = True
            threading.Thread(target=self._timer_thread, daemon=True).start()

    def on_create_record_done(self, *args, **kwargs):
        data = kwargs.get('data')
        self.record_id = data['uuid']
        print(f'REC Grabacion iniciada - ID: {self.record_id}')
        self.recording = True
        threading.Thread(target=self._timer_thread, daemon=True).start()

    def on_stop_record_done(self, *args, **kwargs):
        data = kwargs.get('data')
        print(f'REC Grabacion detenida: {data["uuid"]}')
        print('REC Esperando post-procesamiento...')
        winsound.Beep(1300, 600)
        winsound.Beep(1900, 50)
        winsound.Beep(1900, 50)

    def on_post_processing_done(self, *args, **kwargs):
        record_id = kwargs.get('data')
        print(f'REC Post-procesamiento listo - exportando a {self.export_folder}')
        self.c.export_record(
            self.export_folder,
            self.export_data_types,
            self.export_format,
            [record_id],
            self.export_version
        )

    def on_export_record_done(self, *args, **kwargs):
        data = kwargs.get('data')
        print(f'OK Exportacion completada: {data}')

    def on_new_pow_data(self, *args, **kwargs):
        data = kwargs.get('data')
        if not data or 'pow' not in data:
            return
        try:
            arr = np.array(data['pow']).reshape(len(SENSORS), len(BANDS))

            # Bufer para las graficas
            with self.lock:
                for i, sensor in enumerate(SENSORS):
                    for j, band in enumerate(BANDS):
                        self.buffers[sensor][band].append(arr[i, j])

            row = []
            for col in self.feature_cols:
                _, sensor, band = col.split('.')
                row.append(arr[self.sensor_idx[sensor], self.band_idx[band]])

            # Pasar como DataFrame con nombres de columna 
            X_live        = pd.DataFrame([row], columns=self.feature_cols)
            X_live_scaled = self.scaler.transform(X_live)
            pred          = int(self.model.predict(X_live_scaled)[0])

            texto, color = NIVELES[pred]
            with self.pred_lock:
                self.nivel_actual = pred
                self.nivel_texto  = texto
                self.nivel_color  = color

        except Exception as e:
            print(f'Error: procesando datos: {e}')

    def on_inform_error(self, *args, **kwargs):
        print(f'CORTEX ERROR {kwargs.get("error_data")}')

    #  Temporizador 
    def _timer_thread(self):
        print(f'REC Grabando durante {self.record_duration_s} segundos...')
        while self.elapsed < self.record_duration_s:
            time.sleep(1)
            self.elapsed += 1
            remaining = self.record_duration_s - self.elapsed
            print(f'REC {self.elapsed:>3}s / {self.record_duration_s}s  '
                  f'(restan {remaining}s)', end='\r')
        print()
        self.recording = False

        if self.guardar_datos:
            # Mod GRABACION 
            print('REC Tiempo completado - deteniendo grabacion...')
            self.c.stop_record()
        else:
            # Mod VISUALIZACION
            print('VIS Tiempo completado - cerrando visualizacion...')

    #  Arranque 
    def start(self):
        print('Iniciando conexion con el headset...')
        threading.Thread(target=self.c.open, daemon=True).start()
        self._launch_plot()

    #  Grafica en tiempo real 
    def _launch_plot(self):
        fig, axes = plt.subplots(
            len(DISPLAY_SENSORS), 1,
            figsize=(14, len(SENSORS) * 1.6),
            sharex=True
        )
        fig.patch.set_facecolor(BG)

        self.title_obj = fig.suptitle(
            f"EEG Band Powers  |  {self.record_title}  |  Esperando sesion...",
            color="black", fontsize=22, fontweight="bold", y=1.002
        )

        lines = {}
        x = np.arange(HISTORY)

        for ax, sensor in zip(axes, DISPLAY_SENSORS):
            ax.set_facecolor(BG2)
            ax.set_ylabel(sensor, color="black", fontsize=7,
                          rotation=0, labelpad=28, va="center")
            ax.tick_params(colors="black", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.set_xlim(0, HISTORY - 1)
            ax.set_ylim(0, 1)

            lines[sensor] = {}
            for band, color in zip(BANDS, BAND_COLORS):
                line, = ax.plot(x, [0]*HISTORY, color=color,
                                linewidth=0.9, label=band)
                lines[sensor][band] = line

        axes[-1].legend(
            loc="upper left", fontsize=6,
            facecolor=BG2, labelcolor="black",
            edgecolor=BORDER, ncol=5
        )
        axes[-1].set_xlabel("Muestras", color="black", fontsize=8)
        plt.tight_layout()

        self._overlay = fig.text(
            0.5, 0.5, '',
            ha='center', va='center',
            fontsize=28, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=1', facecolor='#1A1F2E',
                      edgecolor=ACCENT, linewidth=3),
            zorder=10, visible=False
        )
        self._closing       = False
        self._close_tick    = 0
        self._close_seconds = 8

        def update(_frame):

            #  modo cierre
            if self._closing:
                if self.guardar_datos:
                    close_text = f'Guardando datos:\n{self.export_folder}\n\nCerrando en'
                else:
                    close_text = 'Sesion finalizada\n\nCerrando en'

                secs_left = max(0, self._close_seconds - int(self._close_tick * 0.1))
                self._overlay.set_text(f'{close_text} {secs_left}s...')
                self._close_tick += 1
                if self._close_tick >= self._close_seconds * 10:
                    plt.close(fig)
                return []

            #  titulo para cronometro y nivel de estres
            if self.recording:
                remaining = self.record_duration_s - self.elapsed
                mins,  secs  = divmod(self.elapsed, 60)
                rmins, rsecs = divmod(remaining, 60)

                with self.pred_lock:
                    nivel_txt   = self.nivel_texto
                    nivel_color = self.nivel_color

                # grabando y visualizando
                modo_str = "GRABANDO" if self.guardar_datos else "VISUALIZANDO"

                status = (f"{modo_str}  {mins:02d}:{secs:02d} / "
                          f"{self.record_duration_s//60:02d}:{self.record_duration_s%60:02d}"
                          f"  (restan {rmins:02d}:{rsecs:02d})"
                          f"  |  ESTRÉS: {nivel_txt}")
                self.title_obj.set_color(nivel_color)

            elif self.session_active and self.elapsed >= self.record_duration_s:
                if self.guardar_datos:
                    status = "Exportando datos..."
                else:
                    status = "Sesion finalizada"
                self.title_obj.set_color("#F4A261")
                if not self._closing:
                    self._closing = True
                    self._overlay.set_visible(True)
                    for ax in axes:
                        ax.patch.set_alpha(0.4)
            else:
                status = "Esperando sesion..."
                self.title_obj.set_color("white")

            self.title_obj.set_text(
                f"EEG Band Powers  |  {self.record_title}  |  {status}"
            )

            with self.lock:
                for sensor in DISPLAY_SENSORS:
                    y_all = np.array([list(self.buffers[sensor][b]) for b in BANDS])
                    vmin  = y_all.min()
                    vmax  = y_all.max() if y_all.max() != 0 else 1.0
                    idx   = DISPLAY_SENSORS.index(sensor)
                    axes[idx].set_ylim(
                        vmin - 0.05 * abs(vmax),
                        vmax + 0.15 * abs(vmax)
                    )
                    for band in BANDS:
                        lines[sensor][band].set_ydata(
                            list(self.buffers[sensor][band])
                        )
            return [lines[s][b] for s in DISPLAY_SENSORS for b in BANDS]

        self._ani = animation.FuncAnimation(
            fig, update, interval=100, blit=False, cache_frame_data=False
        )
        plt.show()
        sys.exit(0)

#  Main 
def main():
    client_id     = 'yVJVw4Q44g5uVsCPmb1yGf8N0UTfIGbWdLeEHrJM'
    client_secret = ('Xp27kK3CzWFXycz4V4ioJRiTUqAFFxFtCPyHRowmmmKn3xyeZsulw0GC3HyTIxBjjnwC7D76uWUADQAqIi0ygeRBFaLNm8EYQNEJVttU3OwBQmDjYPscFWEV1kSk37Cc')

    launcher = Launcher()

    if launcher.result is None:
        print('Cancelado por el usuario.')
        sys.exit(0)

    cfg = launcher.result
    print(f'Iniciando con: {cfg}')

    monitor = MonitorEEG(
        client_id, client_secret,
        record_title       = cfg['record_title'],
        record_description = cfg['record_description'],
        record_duration_s  = cfg['record_duration_s'],
        export_folder      = cfg['export_folder'],
        export_data_types  = ['BP'],
        export_format      = 'CSV',
        export_version     = 'V2',
        guardar_datos      = cfg['guardar_datos'],
    )
    monitor.start()

if __name__ == '__main__':
    main()