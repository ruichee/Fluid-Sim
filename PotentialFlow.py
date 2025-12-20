import numpy as np
import matplotlib.pyplot as plt


class ElementaryFlow:

    def __init__(self):
        pass


# 1D arrays
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)

# Meshgrid
x, y = np.meshgrid(X, Y)

# Assign vector directions
K = 3
U = 1
u = -K*(y/x**2) / (1 + (y/x)**2) + U
v = K*(1/x) / (1 + (y/x)**2)

# Depict illustration
plt.figure(figsize=(10, 10))
plt.streamplot(X, Y, u, v, density=1.4, linewidth=None, color="#A00000")
'''plt.plot(-1,0,'-or')
plt.plot(1,0,'-og')'''
plt.title('Potential Flow')

# Show plot with grid
plt.grid()
plt.show()