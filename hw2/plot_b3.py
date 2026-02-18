
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def f_a(x1, x2):
    return x1**2 + x2**2

def f_b(x1, x2):
    return -x1 * np.log(x1) - x2 * np.log(x2)

def f_c(x1, x2):
    return np.abs(x1) + np.abs(x2)

def plot_surface(ax3d, title, X1, X2, Z):
    ax3d.plot_surface(X1, X2, Z, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax3d.set_title(title)
    ax3d.set_xlabel("$x_1$")
    ax3d.set_ylabel("$x_2$")
    ax3d.set_zlabel("$f(x_1,x_2)$")
    ax3d.view_init(elev=25, azim=-55)
    ax3d.xaxis.set_major_locator(MaxNLocator(4))
    ax3d.yaxis.set_major_locator(MaxNLocator(4))
    ax3d.zaxis.set_major_locator(MaxNLocator(4))

def main():
    fig = plt.figure(figsize=(18, 5))

    # (a)
    xa = np.linspace(-2, 2, 200)
    X1a, X2a = np.meshgrid(xa, xa)
    Za = f_a(X1a, X2a)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plot_surface(ax1, "(a) $f(x) = x_1^2 + x_2^2$", X1a, X2a, Za)

    # (b)
    xb = np.linspace(0.05, 2.0, 200)
    X1b, X2b = np.meshgrid(xb, xb)
    Zb = f_b(X1b, X2b)
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_surface(ax2, "(b) $f(x) = -x_1 \ln(x_1) - x_2 \ln(x_2)$", X1b, X2b, Zb)

    # (c)
    xc = np.linspace(-2, 2, 200)
    X1c, X2c = np.meshgrid(xc, xc)
    Zc = f_c(X1c, X2c)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_surface(ax3, "(c) $f(x) = |x_1| + |x_2|$", X1c, X2c, Zc)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()