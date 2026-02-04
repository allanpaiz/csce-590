
import numpy as np
import matplotlib.pyplot as plt


a, b = 16.0, 14.0
xg = np.linspace(0, 18, 1400)
yg = np.linspace(0, 18, 1400)
X, Y = np.meshgrid(xg, yg)

feasible = (
    ((X - 8)**2 + (Y - 9)**2 <= 49) &
    (X + Y <= 24) &
    (X >= 2) & (X <= 13)
)

F = (X - a)**2 + (Y - b)**2
F_feas = np.where(feasible, F, np.nan)
idx = np.nanargmin(F_feas)
iy, ix = np.unravel_index(idx, F_feas.shape)
r_star = np.sqrt(F_feas[iy, ix])
x_sol, y_sol = 13.0, 11.0

fig, ax = plt.subplots(figsize=(8, 6))

Z = np.where(feasible, 1.0, np.nan)
ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=["orange"], alpha=0.45)

t = np.linspace(0, 2*np.pi, 800)

xc = 8 + 7*np.cos(t)
yc = 9 + 7*np.sin(t)
ax.plot(xc, yc, color="green")

x_line = np.linspace(5, 18, 500)
y_line = 24 - x_line
ax.plot(x_line, y_line, color="green")
ax.plot([2, 2], [1, 17.5], color="green")
ax.plot([13, 13], [1, 17.5], color="green")

xo = a + r_star * np.cos(t)
yo = b + r_star * np.sin(t)
ax.plot(xo, yo, linestyle="--", color="orange", linewidth=2)
ax.plot(a, b, marker=".", markersize=8, color="orange")
ax.text(a + 0.2, b + 0.2, f"({a:g}, {b:g})", color="orange")
ax.plot(
    x_sol, y_sol,
    marker="o", markersize=7,
    markerfacecolor="g", markeredgecolor="k",
    linewidth=1.5
)
ax.text(x_sol + 1.5, y_sol + 0.2, "Solution:\n x=13, y=11", color="b")
ax.text(7.25, 8.5, "Feasible\nRegion", color="k")
ax.set_aspect("equal", adjustable="box")

ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Example 7 - Constrained Minimization")
ax.grid(False)

plt.show()