import numpy as np
import json

# =========================================================
# Función y gradiente con estabilización numérica (log-sum-exp)
# =========================================================
def f(x):
    x1, x2 = x
    # Evitar overflow: usar log-sum-exp
    a = x1**2 + x2**2
    b = x1 + np.log(10)
    M = max(a, b)
    return M + np.log(np.exp(a - M) + np.exp(b - M))

def grad_f(x):
    x1, x2 = x
    a = x1**2 + x2**2
    b = x1 + np.log(10)
    M = max(a, b)
    # pesos numéricamente estables
    wa = np.exp(a - M)
    wb = np.exp(b - M)
    denom = wa + wb
    # gradiente derivado de la forma log-sum-exp
    dfdx1 = (2 * x1 * wa + wb) / denom
    dfdx2 = (2 * x2 * wa) / denom
    return np.array([dfdx1, dfdx2])

# =========================================================
# Búsqueda de línea de Armijo
# =========================================================
def line_search_armijo(f, grad_f, xk, pk, alpha0=1.0, c=1e-4, rho=0.5):
    alpha = alpha0
    fk = f(xk)
    gradk = grad_f(xk)
    while f(xk + alpha * pk) > fk + c * alpha * np.dot(gradk, pk):
        alpha *= rho
        if alpha < 1e-10:
            break
    return alpha

# =========================================================
# Método BFGS
# =========================================================
def bfgs(f, grad_f, x0, tol=1e-6, max_iter=1000):
    xk = x0.copy()
    n = len(xk)
    Hk = np.eye(n)
    fk = f(xk)
    gk = grad_f(xk)
    iter_data = []

    for k in range(1, max_iter + 1):
        pk = -Hk.dot(gk)
        alpha = line_search_armijo(f, grad_f, xk, pk)
        x_new = xk + alpha * pk
        g_new = grad_f(x_new)
        s = x_new - xk
        y = g_new - gk

        if np.dot(y, s) > 1e-10:
            rho = 1.0 / np.dot(y, s)
            I = np.eye(n)
            Hk = (I - rho * np.outer(s, y)) @ Hk @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        xk, gk, fk = x_new, g_new, f(x_new)
        iter_data.append({
            "iter": k,
            "x": xk.tolist(),
            "f": float(fk),
            "grad_norm": float(np.linalg.norm(gk)),
            "alpha": float(alpha)
        })

        if np.linalg.norm(gk) < tol:
            break

    return xk, fk, np.linalg.norm(gk), k, iter_data

# =========================================================
# Probar en 100 puntos aleatorios
# =========================================================
if __name__ == "__main__":
    import json

    # Cargar puntos iniciales previamente guardados
    with open("initial_points.json", "r") as f_in:
        points = np.array(json.load(f_in))

    results = []
    for i, x0 in enumerate(points):
        try:
            x_opt, f_opt, grad_norm, iters, history = bfgs(f, grad_f, x0)
            results.append({
                "id": i + 1,
                "x0": x0.tolist(),
                "x_opt": x_opt.tolist(),
                "f_opt": float(f_opt),
                "grad_norm": float(grad_norm),
                "iters": int(iters),
                "success": True
            })
        except Exception as e:
            results.append({
                "id": i + 1,
                "x0": x0.tolist(),
                "error": str(e),
                "success": False
            })

    with open("bfgs_results.json", "w") as f_out:
        json.dump(results, f_out, indent=4)

    print("✅ Resultados BFGS guardados en 'bfgs_results.json'")