import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x):
    # f(x) = 5x1^2 + x2^2 + 4x1x2 - 14x1 - 6x2 + 20
    return ((5 * (x[0] ** 2)) + (x[1] ** 2) + (4 * x[0] * x[1]) - (14 * x[0]) - (6 * x[1]) + 20)

def grad_f(x):
    # ∇f(x) = [10x1 + 4x2 - 14, 2x2 + 4x1 - 6]
    return np.array([10 * x[0] + 4 * x[1] - 14, 2 * x[1] + 4 * x[0] - 6])

def find_stepsize(d):
    # Find the stepsize αk that minimizes f(xk + αk * dk)
    alpha = ((d[0] ** 2) + (d[1] ** 2)) / (2 * (5 * (d[0] ** 2) + (d[1] ** 2) + 4 * (d[0] * d[1])))
    return alpha


# Gradient descent loop 
def gradient_descent(x0, epsilon=1e-6, max_iter=100000):
    x = np.asarray(x0, dtype=float).ravel()

    xs = [x.copy()]
    fs = [f(x)]

    k = 0
    print("k | x1k | x2k | d1k | d2k | norm(dk)2 | αk | f(xk)")
    while k < max_iter:
        d = -grad_f(x)
        norm_d = np.linalg.norm(d)

        if norm_d < epsilon:
            print("Converged: ||dk|| < ε")
            print(
                f"{k} | {x[0]:.6f} | {x[1]:.6f} | {d[0]:.6f} | {d[1]:.6f} | "
                f"{norm_d:.6f} | -- | {f(x):.6f}"
            )
            break

        alpha = find_stepsize(d)
        xk = x.copy()
        print(
            f"{k} | {xk[0]:.6f} | {xk[1]:.6f} | {d[0]:.6f} | {d[1]:.6f} | "
            f"{norm_d:.6f} | {alpha:.4f} | {f(xk):.6f}"
        )

        x = x + alpha * d
        xs.append(x.copy())
        fs.append(f(x))
        k += 1

    return np.array(xs), np.array(fs)


# Plotting modified from HW3 - MATH528 - Dr. Colebank
def main():
    # Initial guess
    x0 = np.array([0.0, 10.0])

    # Run gradient descent and record trajectory
    xvals, fvals = gradient_descent(x0, epsilon=1e-6)

    # 3D surface domain 
    x1_space = np.linspace(-10, 10, 400)
    x2_space = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1_space, x2_space)

    # Vectorized evaluation of f on the grid
    Z = 5 * (X1**2) + (X2**2) + 4 * X1 * X2 - 14 * X1 - 6 * X2 + 20

    # Trajectory z-values
    traj_z = fvals

    # 2D contour
    x1c = np.linspace(-5, 5, 400)
    x2c = np.linspace(-2, 10, 400)
    C1, C2 = np.meshgrid(x1c, x2c)
    CZ = 5 * (C1**2) + (C2**2) + 4 * C1 * C2 - 14 * C1 - 6 * C2 + 20


    fig = plt.figure(figsize=(12, 5))


    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax3d.plot(
        xvals[:, 0],
        xvals[:, 1],
        traj_z,
        "--o",
        color="k",
        markersize=4,
        linewidth=1.5,
    )
    ax3d.set_xlabel(r"$x_1$")
    ax3d.set_ylabel(r"$x_2$")
    ax3d.set_zlabel(r"$f(x_1,x_2)$")
    ax3d.set_zlim(0, 1300)
    ax3d.set_title("Surface and optimization trajectory")

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.contour(C1, C2, CZ, levels=30)
    ax2d.plot(xvals[:, 0], xvals[:, 1], "--o", color="k", markersize=4, linewidth=1.5)
    ax2d.set_xlim(-5, 5)
    ax2d.set_ylim(-2, 10)
    ax2d.set_xlabel(r"$x_1$")
    ax2d.set_ylabel(r"$x_2$")
    ax2d.set_title("Level sets and optimization trajectory")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()