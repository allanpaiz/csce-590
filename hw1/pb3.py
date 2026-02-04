
import numpy as np
import matplotlib.pyplot as plt

def f(x):   return 3*x**4 - 4*x**3 + 1
def fp(x):  return 12*x**3 - 12*x**2
def fpp(x): return 36*x**2 - 24*x

x_hpi = 0.0
y_hpi = f(x_hpi)

x_inf2 = 2/3
y_inf2 = f(x_inf2)

x = np.linspace(-1.5, 2.0, 1200)
y = f(x)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

ax.plot(x, y, linewidth=2, label=r"$f(x)=3x^4-4x^3+1$")

# Inflection point
ax.scatter([x_hpi], [y_hpi], s=70, zorder=5, label="Horizontal inflection")
ax.axhline(y_hpi, linestyle="--", linewidth=1.6)
ax.axvline(x_hpi, linestyle=":", linewidth=1.2)

ax.annotate(
    r"Horizontal point of inflection" "\n" r"$(0,1)$",
    xy=(x_hpi, y_hpi),
    xytext=(0.25, 7.5),
    arrowprops=dict(arrowstyle="->", linewidth=1.2),
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", alpha=0.9),
)

ax.set_title(r"Graph of $f(x)=3x^4-4x^3+1$", pad=12)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_xlim(x.min(), x.max())
ax.margins(x=0.02, y=0.08)
ax.legend(frameon=True)

plt.tight_layout()
plt.show()