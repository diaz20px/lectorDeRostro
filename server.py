from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# ==========================================
# CLASIFICADOR
# ==========================================
clasificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# CARGAR USUARIOS
# ==========================================
usuarios = []
ruta_rostros = "rostros"

def cargar_usuarios():
    global usuarios
    usuarios = []

    if not os.path.exists(ruta_rostros):
        os.makedirs(ruta_rostros)

    for archivo in os.listdir(ruta_rostros):

        ruta_imagen = os.path.join(ruta_rostros, archivo)

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            continue

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        rostros = clasificador.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5
        )

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

cargar_usuarios()

# ==========================================
# RECONOCIMIENTO
# ==========================================
@app.route("/reconocer", methods=["POST"])
def reconocer():

    try:

        data = request.json

        imagen_base64 = data["imagen"]

        # ==========================================
        # CONVERTIR BASE64 -> IMAGEN
        # ==========================================
        imagen_bytes = base64.b64decode(imagen_base64)

        np_arr = np.frombuffer(imagen_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "error": "Imagen inválida"
            }), 400

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rostros = clasificador.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # ==========================================
        # RESPUESTA POR DEFECTO
        # ==========================================
        resultado = {
            "detectado": False,
            "nombre": "Desconocido",
            "confianza": "0.00%",
            "acceso": False
        }

        if len(rostros) == 0:
            return jsonify(resultado)

        # SOLO PRIMER ROSTRO
        (x, y, w, h) = rostros[0]

        rostro_actual = gris[y:y+h, x:x+w]

        rostro_actual = cv2.resize(rostro_actual, (200, 200))

        mejor_nombre = "Desconocido"

        menor_diferencia = 999999

        # ==========================================
        # COMPARAR CON USUARIOS
        # ==========================================
        for usuario in usuarios:

            diferencia = cv2.absdiff(
                usuario["rostro"],
                rostro_actual
            )

            resultado_diff = np.mean(diferencia)

            if resultado_diff < menor_diferencia:

                menor_diferencia = resultado_diff

                mejor_nombre = usuario["nombre"]

        # ==========================================
        # CALCULAR CONFIANZA
        # ==========================================
        UMBRAL = 50

        if menor_diferencia < UMBRAL:

            confianza = max(
                0.0,
                min(
                    100.0,
                    (1 - (menor_diferencia / UMBRAL)) * 100
                )
            )

            resultado = {
                "detectado": True,
                "nombre": mejor_nombre,
                "confianza": f"{confianza:.2f}%",
                "acceso": True
            }

        else:

            resultado = {
                "detectado": True,
                "nombre": "Desconocido",
                "confianza": "0.00%",
                "acceso": False
            }

        return jsonify(resultado)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# ==========================================
# INICIAR SERVIDOR
# ==========================================
if __name__ == "__main__":

    print("===================================")
    print(" SERVIDOR BIOMÉTRICO INICIADO ")
    print("===================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )