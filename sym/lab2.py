from typing import Callable, List, Dict, Any
import math
from pprint import pprint


def secant_method(
                    f: Callable[[float], float],
                    x0: float,
                    x1: float,
                    eps: float = 1e-6,
                    max_iter: int = 10_000,
                    ) -> Dict[str, Any]:

    history: List[Dict[str, float]] = []
    x_prev, x_curr = x0, x1
    f_prev, f_curr = f(x_prev), f(x_curr)

    for i in range(1, max_iter + 1):
        denom = f_curr - f_prev
        if denom == 0:
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        f_next = f(x_next)

        history.append(
            {
                "iteration": i,
                "x_prev": x_prev,
                "x_curr": x_curr,
                "x_next": x_next,
                "f_x_prev": f_prev,
                "f_x_curr": f_curr,
                "f_x_next": f_next,
            }
        )

        if abs(f_next) <= eps and abs(x_next - x_curr) <= eps:
            break

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return {
        "root": x_curr,
        "f_root": f_curr,
        "iterations": len(history),
        "history": history,
    }


def func_1(x: float) -> float:
    return x ** 3 - x - 2


def func_2(x: float) -> float:
    return math.cos(x) - x


if __name__ == "__main__":
    result1 = secant_method(func_1, 1, 2, eps=1e-8)
    print("Result 1:")
    pprint(result1, width=80, compact=True)

    result2 = secant_method(func_2, 0, 1, eps=1e-8)
    print("\nResult 2:")
    pprint(result2, width=80, compact=True)
