import os
import sys
import cv2
import numpy as np
from joblib import load

# Cargar el modelo una sola vez
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_dispositivo.pkl")
modelo = load(ruta_modelo)

def predecir(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        return "ERROR: No se pudo cargar la imagen"
    imagen = cv2.resize(imagen, (128, 128))
    vector = imagen.flatten().reshape(1, -1)
    prediccion = modelo.predict(vector)[0]
    return str(prediccion)

# Escuchar entradas
for linea in sys.stdin:
    ruta = linea.strip()
    if not ruta:
        continue
    if ruta.lower() == "salir":
        break
    resultado = predecir(ruta)
    print("Prediccion:" + resultado)
    sys.stdout.flush()
