import numpy as np
import matplotlib.pyplot as plt

def x2_on_curve(x1):
    return (3.0 - x1**2) / 3.0

def ellipse_residual(x1, x2):
    return 4.0 * x1**2 + 9.0 * x2**2 - 36.0

x1_left = -1.0

def g(x1):
    return ellipse_residual(x1, x2_on_curve(x1))

a, b = 2.4, 2.6
ga, gb = g(a), g(b)
assert ga < 0 and gb > 0, "Failed to bracket the right endpoint; adjust [a,b]."

for _ in range(80):
    m = 0.5 * (a + b)
    gm = g(m)
    if ga * gm <= 0:
        b, gb = m, gm
    else:
        a, ga = m, gm

x1_right = 0.5 * (a + b)
coeff = [2.0, 0.0, 21.0, -27.0]
roots = np.roots(coeff)
real_roots = roots[np.isclose(roots.imag, 0.0)].real
x1_star = float(real_roots[0])
x1_star = min(max(x1_star, x1_left), x1_right)
x2_star = x2_on_curve(x1_star)
f_star = (x1_star - 3.0) ** 2 + (x2_star - 3.0) ** 2
r_star = np.sqrt(f_star)

# Plot
x1 = np.linspace(-4, 4, 900)
x2 = np.linspace(-4, 4, 900)
X1, X2 = np.meshgrid(x1, x2)
inside_ellipse = (4.0 * X1**2 + 9.0 * X2**2) <= 36.0

fig, ax = plt.subplots(figsize=(8, 6))

ax.contourf(X1, X2, inside_ellipse.astype(float), levels=[0.5, 1.5], alpha=0.15)

# Ellipse
t = np.linspace(0, 2 * np.pi, 600)
xe = 3.0 * np.cos(t)
ye = 2.0 * np.sin(t)
ax.plot(xe, ye, linewidth=2)

# Parabola
x1_curve = np.linspace(-4.0, 4.0, 800)
x2_curve = x2_on_curve(x1_curve)
ax.plot(x1_curve, x2_curve, linestyle="--", linewidth=1.5)

# Feasible Set
x1_feas = np.linspace(x1_left, x1_right, 500)
x2_feas = x2_on_curve(x1_feas)
ax.plot(x1_feas, x2_feas, linewidth=3, label="Feasible set")
ax.axvline(-1.0, linestyle=":", linewidth=2)

# Objective center and level set
ax.plot(3.0, 3.0, marker="x", markersize=9, linewidth=0)
xo = 3.0 + r_star * np.cos(t)
yo = 3.0 + r_star * np.sin(t)
ax.plot(xo, yo, linestyle=":", linewidth=1.5)
ax.plot(x1_star, x2_star, marker="o", markersize=8, linewidth=0)
ax.plot([3.0, x1_star], [3.0, x2_star], linestyle=":", linewidth=1.5, color="black")

# Labels
ax.text(-0.5, -1.0, r'$4x_1^2 + 9x_2^2 = 36$', fontsize=10)
ax.text(-3.0, -2.4, r'$x_1^2 + 3x_2 = 3$', fontsize=10)
ax.text(-2.1, 2.5, r'$x_1 = -1$', fontsize=10)
ax.text(3.1, 3.1, r'(3,3)', fontsize=10)
ax.text(x1_star + 0.3, x2_star + 0.1, r'$x^*$', fontsize=12)


ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Feasible Set")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", fontsize=9)

plt.show()

print(f"Feasible x1 interval: [{x1_left:.10f}, {x1_right:.10f}]")
print(f"Optimum: x* = ({x1_star:.10f}, {x2_star:.10f})")
print(f"Objective value: f* = {f_star:.10f}")
