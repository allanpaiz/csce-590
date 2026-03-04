import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


EPSILON = 1e-6
MAX_ITER = 100000

X1_MIN, X1_MAX = -5, 5
X2_MIN, X2_MAX = -2, 10


def f_quadratic(x, Q, C):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    Q = np.asarray(Q, dtype=float)
    C = np.asarray(C, dtype=float).reshape(-1, 1)
    return float(0.5 * (x.T @ Q @ x) + (C.T @ x) + 10.0)


def grad_quadratic(x, Q, C):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    Q = np.asarray(Q, dtype=float)
    C = np.asarray(C, dtype=float).reshape(-1, 1)
    # For symmetric Q: ∇f = Qx + C
    return (Q @ x + C).ravel()


def find_stepsize_exact(d, Q):
    d = np.asarray(d, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    denom = float(d.T @ Q @ d)
    if denom <= 0:
        return 0.0
    return float((d.T @ d) / denom)


def gradient_descent_quadratic(x0, Q, C, epsilon=EPSILON, max_iter=MAX_ITER, label=""):
    x = np.asarray(x0, dtype=float).ravel()

    xs = [x.copy()]
    fs = [f_quadratic(x, Q, C)]

    k = 0
    print(f"\n=== Gradient Descent: {label} ===")
    print("k | x1k | x2k | d1k | d2k | norm(dk)2 | αk | f(xk)")
    while k < max_iter:
        d = -grad_quadratic(x, Q, C)
        norm_d = np.linalg.norm(d)

        if norm_d < epsilon:
            print("Converged: ||dk|| < ε")
            print(
                f"{k} | {x[0]:.6f} | {x[1]:.6f} | {d[0]:.6f} | {d[1]:.6f} | "
                f"{norm_d:.6f} | -- | {f_quadratic(x, Q, C):.6f}"
            )
            break

        alpha = find_stepsize_exact(d, Q)
        xk = x.copy()
        print(
            f"{k} | {xk[0]:.6f} | {xk[1]:.6f} | {d[0]:.6f} | {d[1]:.6f} | "
            f"{norm_d:.6f} | {alpha:.6f} | {f_quadratic(xk, Q, C):.6f}"
        )

        x = x + alpha * d
        xs.append(x.copy())
        fs.append(f_quadratic(x, Q, C))
        k += 1

    return np.array(xs), np.array(fs)


def make_surface_grid(Q, C, x1_range=(-10, 10), x2_range=(-10, 10), n=300):
    x1 = np.linspace(x1_range[0], x1_range[1], n)
    x2 = np.linspace(x2_range[0], x2_range[1], n)
    X1, X2 = np.meshgrid(x1, x2)

    q11, q12 = Q[0, 0], Q[0, 1]
    q21, q22 = Q[1, 0], Q[1, 1]
    c1, c2 = C[0], C[1]

    Z = 0.5 * (q11 * X1**2 + (q12 + q21) * X1 * X2 + q22 * X2**2) + c1 * X1 + c2 * X2 + 10
    return X1, X2, Z


def plot_case(xs, fs, Q, C, title_prefix):
    # 3D surface grid
    X1, X2, Z = make_surface_grid(Q, C, x1_range=(-10, 10), x2_range=(-10, 10), n=300)

    # 2D contour grid
    C1, C2, CZ = make_surface_grid(Q, C, x1_range=(X1_MIN, X1_MAX), x2_range=(X2_MIN, X2_MAX), n=400)

    fig = plt.figure(figsize=(12, 5))

    # 3D surface + trajectory
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax3d.plot(xs[:, 0], xs[:, 1], fs, "--o", color="k", markersize=4, linewidth=1.5)
    ax3d.set_xlabel(r"$x_1$")
    ax3d.set_ylabel(r"$x_2$")
    ax3d.set_zlabel(r"$f(x)$")
    ax3d.set_title(f"{title_prefix}: Surface + trajectory")

    # 2D contour + trajectory
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.contour(C1, C2, CZ, levels=30)
    ax2d.plot(xs[:, 0], xs[:, 1], "--o", color="k", markersize=4, linewidth=1.5)
    ax2d.set_xlim(X1_MIN, X1_MAX)
    ax2d.set_ylim(X2_MIN, X2_MAX)
    ax2d.set_xlabel(r"$x_1$")
    ax2d.set_ylabel(r"$x_2$")
    ax2d.set_title(f"{title_prefix}: Level sets + trajectory")

    plt.tight_layout()
    plt.show()


def main():
    x0 = np.array([40.0, -100.0])
    C = np.array([14.0, 6.0])

    # Case 1
    Q1 = np.array([[20.0, 5.0], [5.0, 2.0]])
    xs1, fs1 = gradient_descent_quadratic(
        x0=x0, Q=Q1, C=C, epsilon=EPSILON, label="Case 1: Q=[[20,5],[5,2]]"
    )
    plot_case(xs1, fs1, Q1, C, "Case 1")

    # Case 2
    Q2 = np.array([[20.0, 5.0], [5.0, 16.0]])
    xs2, fs2 = gradient_descent_quadratic(
        x0=x0, Q=Q2, C=C, epsilon=EPSILON, label="Case 2: Q=[[20,5],[5,16]]"
    )
    plot_case(xs2, fs2, Q2, C, "Case 2")


if __name__ == "__main__":
    main()