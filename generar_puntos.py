import numpy as np
import json

# Generar 100 puntos iniciales uniformes en [-100, 100]
np.random.seed(42)
points = np.random.uniform(-100, 100, size=(100, 2))

# Guardarlos como archivo JSON
with open("initial_points.json", "w") as f:
    json.dump(points.tolist(), f, indent=4)

print("✅ Archivo 'initial_points.json' creado con 100 puntos iniciales.")
