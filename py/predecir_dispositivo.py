import sys
import os
import cv2
import numpy as np
from joblib import load


# Leer ruta de imagen desde los argumentos
if len(sys.argv) < 2:
    print("ERROR: Debes pasar la ruta de una imagen")
    sys.exit(1)

ruta_imagen = sys.argv[1]

# Construir ruta absoluta al modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_dispositivo.pkl")

if not os.path.exists(ruta_modelo):
    print("ERROR: No se encontró el modelo en:", ruta_modelo)
    sys.exit(1)

# Cargar modelo
modelo = load(ruta_modelo)

# Leer imagen
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("ERROR: No se pudo cargar la imagen")
    sys.exit(1)

imagen = cv2.resize(imagen, (128, 128))
vector = imagen.flatten().reshape(1, -1)

# Predecir
prediccion = modelo.predict(vector)[0]
print("Prediccion:", prediccion)
