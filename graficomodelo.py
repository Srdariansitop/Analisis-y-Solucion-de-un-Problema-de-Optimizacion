import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función estable
def f_stable(x):
    x1, x2 = x
    a = x1**2 + x2**2
    b = x1 + np.log(10.0)
    M = np.maximum(a, b)
    return M + np.log(np.exp(a - M) + np.exp(b - M))

# Crear una malla de puntos
x1 = np.linspace(-4, 2, 200)
x2 = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1, x2)

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f_stable([X1[i, j], X2[i, j]])

# ---- Contour plot ----
plt.figure(figsize=(7, 6))
cp = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
plt.colorbar(cp)
plt.title('Contornos de f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ---- Superficie 3D ----
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', linewidth=0, alpha=0.8)
ax.set_title('Superficie de f(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()
