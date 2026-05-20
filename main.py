import cv2
import os
import numpy as np
import customtkinter as ctk
import threading
from PIL import Image, ImageTk
from tkinter import messagebox

# Configuración del entorno visual exterior
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# ==========================================
# 1. TU CONFIGURACIÓN Y CARGA ORIGINAL
# ==========================================
clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

usuarios = []
ruta_rostros = "rostros"

def cargar_usuarios_sistema():
    global usuarios
    usuarios = []
    if os.path.exists(ruta_rostros):
        for archivo in os.listdir(ruta_rostros):
            ruta_imagen = os.path.join(ruta_rostros, archivo)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                continue
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            rostros = clasificador.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in rostros:
                rostro = gris[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (200, 200))
                nombre = os.path.splitext(archivo)[0]
                usuarios.append({
                    "nombre": nombre,
                    "rostro": rostro
                })
                print(f"Usuario cargado: {nombre}")
                break

cargar_usuarios_sistema()

# ==========================================
# 2. TU BUCLE PRINCIPAL (RENDERIZADO EN LA APP)
# ==========================================
def funcion_reconocimiento_original():
    global app
    app.camara = cv2.VideoCapture(0)

    while app.camara_activa and app.ejecutandose:
        ret, frame = app.camara.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = clasificador.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        mejor_nombre = "Desconocido"
        menor_diferencia = 999999

        # Almacenar temporalmente el cuadro actual y el recorte gris para la captura
        if len(rostros) > 0:
            (x_c, y_c, w_c, h_c) = rostros[0]
            app.ultimo_rostro_gris = gris[y_c:y_c+h_c, x_c:x_c+w_c]
            app.ultimo_frame_bgr = frame.copy() # Guardamos la foto real a color

        for (x, y, w, h) in rostros:
            rostro_actual = gris[y:y+h, x:x+w]
            rostro_actual = cv2.resize(rostro_actual, (200, 200))

            for usuario in usuarios:
                diferencia = cv2.absdiff(usuario["rostro"], rostro_actual)
                resultado = np.mean(diferencia)
                if resultado < menor_diferencia:
                    menor_diferencia = resultado
                    mejor_nombre = usuario["nombre"]

            if menor_diferencia > 50:
                mejor_nombre = "Desconocido"

            # Compartir el nombre actual con la app global
            app.nombre_actual_detectado = mejor_nombre

            # Conexión directa con las etiquetas de la interfaz
            if mejor_nombre != "Desconocido":
                porcentaje_confianza = max(0.0, min(100.0, (1 - (menor_diferencia / 50.0)) * 100))
                
                app.cambiar_datos_interfaz(
                    nombre=mejor_nombre, 
                    id_user="001", 
                    confianza=f"{porcentaje_confianza:.2f}%", 
                    acceso_valido=True
                )
                color_recuadro = (0, 255, 0)
            else:
                app.cambiar_datos_interfaz(
                    nombre="Desconocido", 
                    id_user="---", 
                    confianza="0.00%", 
                    acceso_valido=False
                )
                color_recuadro = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color_recuadro, 2)
            cv2.putText(frame, mejor_nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_recuadro, 2)

        if len(rostros) == 0:
            app.nombre_actual_detectado = "Desconocido"
            app.cambiar_datos_interfaz("Esperando...", "---", "0.00%", None)

        # Renders en el contenedor central móvil de la App
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((620, 440), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        if app.ejecutandose:
            app.lbl_pantalla_video.configure(image=img_tk)
            app.lbl_pantalla_video.image = img_tk

    if app.camara:
        app.camara.release()

# ==========================================
# 3. INTERFAZ GRÁFICA UNIFICADA (TRES BOTONES)
# ==========================================
class VentanaEstilos(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SISTEMA DE RECONOCIMIENTO FACIAL")
        self.geometry("700x820")
        self.resizable(False, False)
        self.configure(fg_color="#000000")
        
        # Variables de control interno
        self.ejecutandose = True
        self.camara_activa = False
        self.camara = None
        self.ultimo_rostro_gris = None
        self.ultimo_frame_bgr = None
        self.nombre_actual_detectado = "Desconocido"

        # Encabezado Superior
        self.lbl_titulo = ctk.CTkLabel(self, text="[ ⿃ ]  SISTEMA DE RECONOCIMIENTO FACIAL", font=("Arial Black", 18), text_color="#00FF00")
        self.lbl_titulo.pack(pady=15)

        # Centro: Contenedor de la cámara integrado
        self.lbl_pantalla_video = ctk.CTkLabel(self, text="CÁMARA APAGADA", font=("Arial Bold", 14), fg_color="#101010", width=620, height=440)
        self.lbl_pantalla_video.pack(pady=5)

        # Panel Dinámico de Estado
        self.frame_alerta = ctk.CTkFrame(self, fg_color="#000000", border_color="#111111", border_width=2, height=50)
        self.frame_alerta.pack(fill="x", padx=40, pady=10)
        self.frame_alerta.pack_propagate(False)
        self.lbl_alerta = ctk.CTkLabel(self.frame_alerta, text="SISTEMA EN ESPERA", font=("Arial Black", 15), text_color="#444444")
        self.lbl_alerta.pack(expand=True)

        # Panel de Datos del Lector
        self.frame_datos = ctk.CTkFrame(self, fg_color="#050505", border_color="#111111", border_width=1, height=90)
        self.frame_datos.pack(fill="x", padx=40, pady=5)
        
        self.lbl_nombre = ctk.CTkLabel(self.frame_datos, text="Esperando...", font=("Arial Black", 20), text_color="#00FF00")
        self.lbl_nombre.place(x=20, y=12)
        self.lbl_id = ctk.CTkLabel(self.frame_datos, text="ID: ---", font=("Arial", 14), text_color="#FFFFFF")
        self.lbl_id.place(x=20, y=48)
        self.lbl_confianza = ctk.CTkLabel(self.frame_datos, text="Confianza: 0.00%", font=("Arial Bold", 16), text_color="#00FF00")
        self.lbl_confianza.place(x=450, y=30)

        # Panel de Control Inferior Modificado: 3 Columnas Distribuidas de manera uniforme
        self.frame_botones = ctk.CTkFrame(self, fg_color="#0a0a0a", height=65)
        self.frame_botones.pack(fill="x", padx=40, pady=15)
        self.frame_botones.grid_columnconfigure((0, 1, 2), weight=1) # Distribución proporcional

        self.btn_iniciar = ctk.CTkButton(self.frame_botones, text="▶  INICIAR", command=self.arrancar_lector, fg_color="#1c1c1c", hover_color="#2b2b2b", height=45, font=("Arial Bold", 13))
        self.btn_iniciar.grid(row=0, column=0, padx=15, pady=10, sticky="ew")
        
        self.btn_detener = ctk.CTkButton(self.frame_botones, text="■  DETENER", command=self.parar_lector, fg_color="#1c1c1c", hover_color="#2b2b2b", text_color="#FF3333", height=45, font=("Arial Bold", 13))
        self.btn_detener.grid(row=0, column=1, padx=15, pady=10, sticky="ew")
        
        self.btn_capturar = ctk.CTkButton(self.frame_botones, text="📷  CAPTURAR", command=self.capturar_rostro, fg_color="#1c1c1c", hover_color="#2b2b2b", text_color="#3399FF", height=45, font=("Arial Bold", 13))
        self.btn_capturar.grid(row=0, column=2, padx=15, pady=10, sticky="ew")

        # Barra de Estado Inferior
        self.lbl_status = ctk.CTkLabel(self, text="Estado: Inactivo", font=("Arial", 11), text_color="#555555")
        self.lbl_status.pack(side="left", padx=50, pady=5)
        self.lbl_contador = ctk.CTkLabel(self, text=f"Usuarios registrados: {len(usuarios)}", font=("Arial", 11), text_color="#FFFFFF")
        self.lbl_contador.pack(side="right", padx=50, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.cerrar_todo)

    # ==========================================
    # LÓGICA DE LAS TRES FUNCIONALIDADES
    # ==========================================

    def arrancar_lector(self):
        """ FUNCIONALIDAD 1: INICIAR """
        if not self.camara_activa:
            self.camara_activa = True
            self.lbl_status.configure(text="Estado: Sistema activo", text_color="#00FF00")
            hilo = threading.Thread(target=funcion_reconocimiento_original)
            hilo.daemon = True
            hilo.start()

    def parar_lector(self):
        """ FUNCIONALIDAD 2: DETENER (Apaga la cámara por completo) """
        self.camara_activa = False
        self.lbl_status.configure(text="Estado: Inactivo", text_color="#555555")
        self.lbl_pantalla_video.configure(image="", text="CÁMARA APAGADA")
        self.cambiar_datos_interfaz("Esperando...", "---", "0.00%", None)

    def capturar_rostro(self):
        """ FUNCIONALIDAD 3: CAPTURAR (Validación estricta de registros) """
        if not self.camara_activa or self.ultimo_frame_bgr is None:
            messagebox.showwarning("Alerta", "La cámara debe estar activa y detectando un rostro.")
            return

        # Verificar si la persona que está al frente está guardada en el sistema
        if self.nombre_actual_detectado == "Desconocido":
            messagebox.showerror("Acceso Denegado", "El usuario no se encuentra registrado en el sistema.")
        else:
            # Si el usuario ya existe, se captura la imagen original a color
            if not os.path.exists("capturas_registro"):
                os.makedirs("capturas_registro")
            
            ruta_foto = os.path.join("capturas_registro", f"registro_{self.nombre_actual_detectado}.jpg")
            cv2.imwrite(ruta_foto, self.ultimo_frame_bgr)
            
            # Mensaje de éxito solicitado y apagado preventivo/limpieza
            messagebox.showinfo("Sistema Biométrico", f"Usuario aceptado: {self.nombre_actual_detectado}\nFoto de registro guardada con éxito.")
            self.parar_lector()

    def cambiar_datos_interfaz(self, nombre, id_user, confianza, acceso_valido):
        self.lbl_nombre.configure(text=nombre)
        self.lbl_id.configure(text=f"ID: {id_user}")
        self.lbl_confianza.configure(text=f"Confianza: {confianza}")
        
        if acceso_valido is True:
            self.frame_alerta.configure(border_color="#00FF00")
            self.lbl_alerta.configure(text="✔  ACCESO CONCEDIDO", text_color="#00FF00")
            self.lbl_nombre.configure(text_color="#00FF00")
            self.lbl_confianza.configure(text_color="#00FF00")
        elif acceso_valido is False:
            self.frame_alerta.configure(border_color="#FF3333")
            self.lbl_alerta.configure(text="❌  ACCESO DENEGADO", text_color="#FF3333")
            self.lbl_nombre.configure(text_color="#FF3333")
            self.lbl_confianza.configure(text_color="#FF3333")
        else:
            self.frame_alerta.configure(border_color="#111111")
            self.lbl_alerta.configure(text="SISTEMA EN ESPERA", text_color="#444444")
            self.lbl_nombre.configure(text_color="#00FF00")
            self.lbl_confianza.configure(text_color="#00FF00")

    def cerrar_todo(self):
        self.camara_activa = False
        self.ejecutandose = False
        self.destroy()

if __name__ == "__main__":
    app = VentanaEstilos()
    app.mainloop()