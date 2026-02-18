import numpy as np
import matplotlib.pyplot as plt

def setup_ax(ax, title):
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1, 2)

# Data
t = np.linspace(-1, 2, 200)
x1_line, x2_line, x3_line = t, t, t
u = np.linspace(-1, 2, 40)
v = np.linspace(-1, 2, 40)
U, V = np.meshgrid(u, v)
W = 1 - U - V
p = np.array([1/3, 1/3, 1/3])

fig = plt.figure(figsize=(18, 8))

# S1
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
setup_ax(ax1, "$S_1: x_1=x_2=x_3$")
ax1.plot(x1_line, x2_line, x3_line)

# S2
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
setup_ax(ax2, "$S_2: x_1+x_2+x_3=1$")
ax2.plot_surface(U, V, W, alpha=0.35, linewidth=0)

# S3
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
setup_ax(ax3, "$S_3 = S_1 \cap S_2$")
ax3.scatter([p[0]], [p[1]], [p[2]], s=50)

plt.tight_layout()
plt.show()
