import numpy as np

# ---------------------------------------------------------
# Definición de la función, gradiente y log-sum-exp seguro
# ---------------------------------------------------------
def f(x):
    # x es un vector [x0, x1]
    x1, x2 = x
    A = np.exp(x1**2 + x2**2) + 10 * np.exp(x1)
    return np.log(A)

def grad_f(x):
    x1, x2 = x
    A = np.exp(x1**2 + x2**2) + 10 * np.exp(x1)
    dfdx1 = (2 * x1 * np.exp(x1**2 + x2**2) + 10 * np.exp(x1)) / A
    dfdx2 = (2 * x2 * np.exp(x1**2 + x2**2)) / A
    return np.array([dfdx1, dfdx2])

# ---------------------------------------------------------
# Búsqueda de línea con condición de Armijo
# ---------------------------------------------------------
def line_search_armijo(f, grad_f, xk, pk, alpha0=1.0, c=1e-4, rho=0.5):
    alpha = alpha0
    fk = f(xk)
    gradk = grad_f(xk)
    while f(xk + alpha * pk) > fk + c * alpha * np.dot(gradk, pk):
        alpha *= rho
        if alpha < 1e-10:
            break
    return alpha

# ---------------------------------------------------------
# Método BFGS
# ---------------------------------------------------------
def bfgs(f, grad_f, x0, tol=1e-6, max_iter=1000):
    xk = x0.copy()
    n = len(xk)
    Hk = np.eye(n)  # matriz de aproximación del inverso del Hessiano
    fk = f(xk)
    gk = grad_f(xk)
    iter_data = [(0, xk.copy(), fk, np.linalg.norm(gk))]

    for k in range(1, max_iter + 1):
        # Dirección de búsqueda
        pk = -Hk.dot(gk)

        # Búsqueda de línea (Armijo)
        alpha = line_search_armijo(f, grad_f, xk, pk)

        # Actualización de x
        x_new = xk + alpha * pk
        g_new = grad_f(x_new)
        s = x_new - xk
        y = g_new - gk

        # Condición de actualización BFGS (evitar divisiones malas)
        if np.dot(y, s) > 1e-10:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            Hk = (I - rho * np.outer(s, y)) @ Hk @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        # Actualizar variables
        xk, gk, fk = x_new, g_new, f(x_new)
        iter_data.append((k, xk.copy(), fk, np.linalg.norm(gk)))

        # Criterios de paro
        if np.linalg.norm(gk) < tol:
            break

    return xk, fk, np.linalg.norm(gk), k, iter_data

# ---------------------------------------------------------
# Ejemplo de ejecución
# ---------------------------------------------------------
if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])   # punto inicial
    x_opt, f_opt, grad_norm, iters, history = bfgs(f, grad_f, x0)

    print("Resultado BFGS:")
    print(f"  x* = {x_opt}")
    print(f"  f(x*) = {f_opt:.6f}")
    print(f"  ||grad|| = {grad_norm:.2e}")
    print(f"  Iteraciones = {iters}")
