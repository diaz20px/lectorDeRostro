import cv2
import numpy as np
import os

print("1. Librerías importadas con éxito...")

# Nueva ruta oficial de MediaPipe para versiones modernas (0.10.30+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

print("2. Módulos de MediaPipe Tasks cargados...")

# Descargar el modelo oficial de Google si no existe en la carpeta
url_modelo = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
nombre_modelo = "face_landmarker.task"

if not os.path.exists(nombre_modelo):
    print("Descargando modelo de Google Face Landmarker...")
    import urllib.request
    urllib.request.urlretrieve(url_modelo, nombre_modelo)

# Configurar las opciones del detector de rostros moderno
opciones = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=nombre_modelo),
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=False,
    num_faces=1
)

# Inicializar el detector usando el nuevo estándar de Google
with vision.FaceLandmarker.create_from_options(opciones) as detector:
    print("3. Detector FaceLandmarker inicializado con éxito!")
    
    # Abrir la cámara nativa de la laptop
    cap = cv2.VideoCapture(0)
    print("4. Cámara abierta. Iniciando bucle (Presiona 'q' para salir)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede leer el flujo de la cámara.")
            break
            
        # Efecto espejo para que sea natural al moverte
        frame = cv2.flip(frame, 1)
        
        # Convertir el cuadro a RGB (MediaPipe moderno no lee BGR de OpenCV)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear el contenedor de imagen corregido usando mp.Image directamente
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Procesar la imagen para buscar los puntos del rostro
        resultado = detector.detect(mp_image)
        
        # Si encuentra la cara, recorre y dibuja los puntos biométricos
        if resultado.face_landmarks:
            for rostro in resultado.face_landmarks:
                for punto in rostro:
                    # Convertir coordenadas decimales (0.0 a 1.0) a los píxeles reales de tu pantalla
                    alto, ancho, _ = frame.shape
                    cx, cy = int(punto.x * ancho), int(punto.y * alto)
                    
                    # Dibujar un pequeño punto verde de 1 píxel por coordenada
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                    
        # Mostrar los resultados en una ventana flotante estándar
        cv2.imshow('Malla Facial Moderna (MediaPipe Tasks)', frame)
        
        # Romper el bucle de video de forma segura al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema cerrado correctamente.")