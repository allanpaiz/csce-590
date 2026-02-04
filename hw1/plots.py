
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 3, 600)

x2_c1 = (3/8) * (4 - x1)
x2_c2 = 2 - x1
x1_c3 = 1.5

X1, X2 = np.meshgrid(np.linspace(0, 3, 800), np.linspace(0, 3, 800))
feasible = (
    (X1 + (8/3) * X2 <= 4) &
    (X1 + X2 <= 2) &
    (2 * X1 <= 3) &
    (X1 >= 0) &
    (X2 >= 0)
)

x2_obj = -2 * x1 + 3.5

plt.figure(figsize=(8, 6))

Z = np.where(feasible, 1.0, np.nan)
plt.contourf(X1, X2, Z, levels=[0.5, 1.5], colors=["green"], alpha=0.35)

plt.plot(x1, x2_c1, color="b")
plt.plot(x1, x2_c2, color="b")
plt.axvline(x=x1_c3, color="b")
plt.plot(x1, x2_obj, linestyle="--", color="b", linewidth=1)

plt.text(0.45, 0.65, "Feasible Region", color="red")
plt.text(1.55, 2.1, r"$x_1 = \frac{3}{2}$", color="red")
plt.text(0.35, 1.75, r"$x_1 + x_2 = 2$", color="red")
plt.text(2.2, .75, r"$x_1 + \frac{8}{3}x_2 = 4$", color="red")

x_min, y_min = 1.5, 0.5
plt.plot(
    x_min, y_min,
    marker="o", markersize=7,
    markerfacecolor="none", markeredgecolor="r",
    linewidth=1.5
)
plt.text(x_min + 0.05, y_min + 0.08, "Minimizer \n(1.5, 0.5)", color="red")

plt.xlim(0, 3)
plt.ylim(0, 3)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Example 3 - Linear Programming")
plt.grid(True)

plt.show()

