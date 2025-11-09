import numpy as np

# -----------------------------
# Evaluación numéricamente estable de f, gradiente y Hessiano
# f(x,y) = ln(e^{x^2+y^2} + 10 e^{x})
# Usamos log-sum-exp para evitar overflow.
# -----------------------------
def f_stable(x):
    x1, x2 = float(x[0]), float(x[1])
    a = x1**2 + x2**2
    b = x1 + np.log(10.0)    # log(10 * e^{x1}) = x1 + ln(10)
    M = max(a, b)
    val = M + np.log(np.exp(a - M) + np.exp(b - M))
    return val

def grad_f_stable(x):
    x1, x2 = float(x[0]), float(x[1])
    a = x1**2 + x2**2
    b = x1 + np.log(10.0)
    M = max(a, b)
    ea = np.exp(a - M)
    eb = np.exp(b - M)
    denom = ea + eb
    # derivadas en escala estable
    dfdx1 = (2 * x1 * ea + eb) / denom
    dfdx2 = (2 * x2 * ea) / denom
    return np.array([dfdx1, dfdx2])

def hess_f_stable(x):
    # Calculamos el Hessiano de f en forma estable (2x2)
    x1, x2 = float(x[0]), float(x[1])
    a = x1**2 + x2**2
    b = x1 + np.log(10.0)
    M = max(a, b)
    ea = np.exp(a - M)
    eb = np.exp(b - M)
    denom = ea + eb

    # Componentes auxiliares (numeradores escalados)
    A1 = 2 * x1 * ea         # proviene de 2 x1 e^a (escalado)
    B1 = eb                  # proviene de 10 e^{x1} (escalado)
    num_g1 = A1 + B1         # numerador de g1 (componente x1 del gradiente)
    # g2 numerador
    num_g2 = 2 * x2 * ea

    # derivadas de A1, B1
    # dA1/dx1 = 2*ea + 2*x1 * d(ea)/dx1, y d(ea)/dx1 = 2 x1 ea
    dA1_dx1 = 2 * ea + 2 * x1 * (2 * x1 * ea)   # 2 ea + 4 x1^2 ea
    dA1_dx2 = 2 * x1 * (2 * x2 * ea)             # 4 x1 x2 ea
    dB1_dx1 = eb
    dB1_dx2 = 0.0

    # derivadas del denom
    ddenom_dx1 = (2 * x1 * ea) + eb
    ddenom_dx2 = (2 * x2 * ea)

    # H_11: d(g1)/dx1 por regla del cociente
    H11 = ((dA1_dx1 + dB1_dx1) * denom - num_g1 * ddenom_dx1) / (denom**2)

    # H_12: d(g1)/dx2
    H12 = ((dA1_dx2 + dB1_dx2) * denom - num_g1 * ddenom_dx2) / (denom**2)

    # H_21: d(g2)/dx1
    # g2 = (2 x2 ea)/denom -> derivada wrt x1:
    # numerator derivative: 2 x2 * d(ea)/dx1 = 2 x2 * (2 x1 ea)
    dnum_g2_dx1 = 2 * x2 * (2 * x1 * ea)
    H21 = (dnum_g2_dx1 * denom - num_g2 * ddenom_dx1) / (denom**2)

    # H_22: d(g2)/dx2
    # derivative numerator: 2 ea + 2 x2 * d(ea)/dx2 = 2 ea + 4 x2^2 ea
    dnum_g2_dx2 = 2 * ea + 4 * x2**2 * ea
    H22 = (dnum_g2_dx2 * denom - num_g2 * ddenom_dx2) / (denom**2)

    H = np.array([[H11, H12],
                  [H21, H22]])
    # forzamos simetría numérica
    H = 0.5 * (H + H.T)
    return H

# -----------------------------
# Backtracking Armijo line search
# -----------------------------
def backtracking_armijo(f, grad, xk, pk, alpha0=1.0, c=1e-4, rho=0.5, max_iters=50):
    alpha = alpha0
    fk = f(xk)
    gk = grad(xk)
    for _ in range(max_iters):
        newx = xk + alpha * pk
        if f(newx) <= fk + c * alpha * np.dot(gk, pk):
            return alpha
        alpha *= rho
    return alpha

# -----------------------------
# Damped Newton / Trust-region style algorithm
# (Levenberg-Marquardt style damping + Armijo backtracking)
# -----------------------------
def damped_newton_trust(f, grad, hess, x0, tol=1e-8, max_iter=200,
                        lambda0=1e-3, lambda_factor_increase=10.0, lambda_factor_decrease=0.1):
    xk = x0.astype(float).copy()
    lamb = lambda0
    history = []
    for k in range(1, max_iter + 1):
        fk = f(xk)
        gk = grad(xk)
        grad_norm = np.linalg.norm(gk)
        history.append((k, xk.copy(), fk, grad_norm, lamb))
        if grad_norm < tol:
            break

        Hk = hess(xk)
        success = False

        # intentos con damping creciente para asegurar PD y dirección de descenso
        for attempt in range(20):
            try:
                H_reg = Hk + lamb * np.eye(len(xk))
                pk = np.linalg.solve(H_reg, -gk)
            except np.linalg.LinAlgError:
                lamb *= lambda_factor_increase
                continue

            # si no es dirección de descenso, aumentamos damping
            if np.dot(gk, pk) >= 0:
                lamb *= lambda_factor_increase
                continue

            # búsqueda de línea Armijo para el paso propuesto
            alpha = backtracking_armijo(f, grad, xk, pk, alpha0=1.0)
            newx = xk + alpha * pk
            newf = f(newx)

            # criterio simple: aceptamos si hay descenso (podemos refinar con ratio tipo trust-region)
            if newf <= fk + 1e-12:
                success = True
                break
            else:
                lamb *= lambda_factor_increase

        if not success:
            # fallback: gradiente con paso pequeño si no se encuentra paso Newton estable
            pk = -gk
            alpha = backtracking_armijo(f, grad, xk, pk, alpha0=1e-3)
            xk = xk + alpha * pk
            lamb *= lambda_factor_increase
            continue

        # actualizar x y reducir damping (recuperar comportamiento Newton)
        xk = newx
        lamb = max(1e-16, lamb * lambda_factor_decrease)

    # valores finales
    fk = f(xk)
    gk = grad(xk)
    history.append(("final", xk.copy(), fk, np.linalg.norm(gk), lamb))
    return xk, fk, np.linalg.norm(gk), k, history

# -----------------------------
# Ejemplo de ejecución (main)
# -----------------------------
if __name__ == "__main__":
    initials = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 2.0]),
        np.array([2.0, 2.0])
    ]

    results = []
    for x0 in initials:
        xopt, fopt, gnorm, its, hist = damped_newton_trust(
            f_stable, grad_f_stable, hess_f_stable, x0
        )
        print("Init:", x0, "-> x*:", np.round(xopt, 6),
              " f*:", np.round(fopt, 8), "||grad||:", np.format_float_scientific(gnorm, 2),
              "iter:", its)
        results.append((x0, xopt, fopt, gnorm, its))

    # Graficar convergencia (norma del gradiente) del último historial como ejemplo
    try:
        import matplotlib.pyplot as plt
        hist = hist  # historial de la última ejecución
        iter_nums = [h[0] for h in hist if isinstance(h[0], int)]
        grad_norms = [h[3] for h in hist if isinstance(h[0], int)]
        if len(iter_nums) > 0:
            plt.semilogy(iter_nums, grad_norms, marker='o')
            plt.xlabel('Iteración')
            plt.ylabel('||grad|| (escala log)')
            plt.title('Convergencia (ejemplo)')
            plt.grid(True)
            plt.show()
    except Exception:
        # si no hay matplotlib, no graficamos pero el algoritmo sigue funcionando
        pass
