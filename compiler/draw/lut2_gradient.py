import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x0 = np.linspace(0, 1, 100)
x1 = np.linspace(0, 1, 100)
X0, X1 = np.meshgrid(x0, x1)
Z = X1-X0

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X0, X1, Z, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_xlabel('$x_0$', fontsize=14)
ax.set_ylabel('$x_1$', fontsize=14)
ax.set_zlabel(r'$\frac{\partial f_2(\mathbf{x}, \mathbf{w})}{w_{\operatorname{idx}(00_{(2)})}}$', fontsize=16)

ax.set_title(r'$\frac{\partial f_2(\mathbf{x}, \mathbf{w})}{w_{\operatorname{idx}(00_{(2)})}} = (1-x_0)(1-x_1)$', fontsize=16, pad=20)

fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)

plt.tight_layout()
plt.savefig("lut2_gradient.png", dpi=300, bbox_inches='tight')
