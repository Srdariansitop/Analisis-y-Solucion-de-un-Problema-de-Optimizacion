import json
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos (asumiendo que los archivos están disponibles)
try:
    with open('initial_points.json', 'r') as f:
        initial_points = json.load(f)
    with open('bfgs_results.json', 'r') as f:
        bfgs_results = json.load(f)
    with open('results_newton.json', 'r') as f:
        newton_data = json.load(f)
        newton_results = newton_data['results']
except FileNotFoundError as e:
    print(f"Error: El archivo no se encontró: {e}")
    raise

# --- Extracción de Datos ---
x0_all = np.array(initial_points)
bfgs_successful = [res for res in bfgs_results if res.get('success', False)]
x_opt_bfgs = np.array([res['x_opt'] for res in bfgs_successful])
x_opt_newton = np.array([res['x_opt'] for res in newton_results])

# Óptimo Global Aproximado
x_global = -0.901226723
y_global = 0.0

# --- Configuración de las Gráficas ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
plt.suptitle('Comparación de Puntos Óptimos Encontrados (BFGS vs. Newton)', fontsize=18)

# --- Gráfico 1: Vista General ---
ax1 = axes[0]
ax1.set_title('Vista General: Puntos Iniciales ($x_0$) y Óptimos ($x_{opt}$)', fontsize=14)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.grid(True, linestyle=':', alpha=0.6)

# 1. Puntos Iniciales Comunes
ax1.scatter(x0_all[:, 0], x0_all[:, 1], color='gray', marker='.', s=10, alpha=0.5, label='Puntos Iniciales ($x_0$)', zorder=1)

# 2. Resultados BFGS
ax1.scatter(x_opt_bfgs[:, 0], x_opt_bfgs[:, 1], color='red', marker='D', s=50, label='Óptimos BFGS (Exitosos)', zorder=3)

# 3. Resultados Newton
ax1.scatter(x_opt_newton[:, 0], x_opt_newton[:, 1], color='blue', marker='o', s=50, label='Óptimos Newton', zorder=3)

# 4. Óptimo Global
ax1.scatter(x_global, y_global, color='black', marker='*', s=300, label='Óptimo Global (Aprox.)', zorder=5)

ax1.legend()


# --- Gráfico 2: Zoom en la Región Óptima con Ejes Ajustados ---
ax2 = axes[1]
ax2.set_title('Zoom en la Región de Convergencia', fontsize=14)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.grid(True, linestyle=':', alpha=0.7)

# --- Ajuste Clave del Zoom ---
# Límite x: 0.0000005 (5e-7) alrededor del óptimo
ax2.set_xlim(x_global - 5e-7, x_global + 5e-7) 
# Límite y: 0.0000001 (1e-7) alrededor de cero
ax2.set_ylim(-1e-7, 1e-7) 
# -----------------------------

# 1. Resultados BFGS (Zoom)
ax2.scatter(x_opt_bfgs[:, 0], x_opt_bfgs[:, 1], color='red', marker='D', s=80, alpha=0.7, label='Óptimos BFGS (Exitosos)', zorder=3)

# 2. Resultados Newton (Zoom)
ax2.scatter(x_opt_newton[:, 0], x_opt_newton[:, 1], color='blue', marker='o', s=80, alpha=0.7, label='Óptimos Newton', zorder=3)

# 3. Óptimo Global (Zoom)
ax2.scatter(x_global, y_global, color='black', marker='*', s=400, label='Óptimo Global (Aprox.)', zorder=5)

ax2.legend(loc='upper right')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()