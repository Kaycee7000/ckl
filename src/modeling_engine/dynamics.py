import numpy as np
from typing import Callable, Sequence

try:
    from scipy.integrate import solve_ivp
except Exception:
    solve_ivp = None


def solve_ode(func: Callable[[float, Sequence[float]], Sequence[float]], y0: Sequence[float], t_span: Sequence[float], t_eval: Sequence[float] = None):
    if solve_ivp is None:
        raise RuntimeError("scipy is required for ODE solving; install scipy")
    sol = solve_ivp(fun=func, t_span=(t_span[0], t_span[-1]), y0=y0, t_eval=t_eval)
    return sol.t, sol.y
