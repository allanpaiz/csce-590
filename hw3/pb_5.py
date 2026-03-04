import numpy as np
import matplotlib.pyplot as plt

# Points
points = {
    r"$x_a$": (0.5, 0.5),
    r"$x_b$": (1.0, 0.0),
    r"$x_c$": (-1.0, 0.0),
    r"$x_d$": (-0.5, 0.0),
    r"$x_e$": (1/np.sqrt(2), 1/np.sqrt(2)),
}

# Plot window
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.6, 1.2

# Grid for feasible region
nx, ny = 800, 700
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Constraints
c1 = 1 - (X**2 + Y**2)
c2 = np.sqrt(2) - (X + Y)
c3 = Y

feasible = (c1 >= 0) & (c2 >= 0) & (c3 >= 0)

# Colors
light_red = (1.0, 0.7, 0.7)
dark_red  = (0.6, 0.0, 0.0)
blue      = (0.0, 0.0, 0.7)
green     = (0.0, 0.55, 0.0)

fig, ax = plt.subplots(figsize=(7.2, 6.4))

# Shade feasible region
ax.contourf(
    X, Y, feasible.astype(int),
    levels=[0.5, 1.5],
    colors=[light_red],
    alpha=0.35
)

# x2 >= 0
t = np.linspace(0, 2*np.pi, 1200)
xc = np.cos(t)
yc = np.sin(t)
mask_upper = yc >= 0
ax.plot(xc[mask_upper], yc[mask_upper],
        color=dark_red, linewidth=2.2)
mask_lower = yc < 0
ax.plot(xc[mask_lower], yc[mask_lower],
        color=dark_red, linewidth=1.0, linestyle=":")


# x2 = 0 , -1 <= x1 <= 1
ax.plot([-1, 1], [0, 0],
        color=dark_red, linewidth=2.2)
ax.plot([x_min, -1], [0, 0],
        color=green, linewidth=1.0, linestyle=":")
ax.plot([1, x_max], [0, 0],
        color=green, linewidth=1.0, linestyle=":")

# x1 + x2 = sqrt(2)
x_line = np.linspace(x_min, x_max, 600)
y_line = np.sqrt(2) - x_line
ax.plot(x_line, y_line,
        color=blue, linewidth=1.0, linestyle=":")

# Plot
for lab, (px, py) in points.items():
    ax.scatter([px], [py], s=55, color="black", zorder=5)
    ax.annotate(
        lab, (px, py),
        textcoords="offset points", xytext=(8, 8),
        fontsize=12
    )
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()