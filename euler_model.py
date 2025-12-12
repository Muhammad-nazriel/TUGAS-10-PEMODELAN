import numpy as np

def logistic_euler(U0, r, K, h, t_end):
    """
    Euler method for logistic growth model.
    dU/dt = rU(1 - U/K)
    """

    t_values = np.arange(0, t_end, h)
    U_values = np.zeros_like(t_values)
    U_values[0] = U0

    for i in range(1, len(t_values)):
        U = U_values[i - 1]
        dUdt = r * U * (1 - U / K)
        U_values[i] = U + h * dUdt

    return t_values, U_values
