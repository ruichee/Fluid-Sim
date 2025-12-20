import numpy as np
import matplotlib.pyplot as plt


class ElementaryFlow:

    def __init__(self):
        self.u: function[float, float]
        self.v: function[float, float]

class Uniform(ElementaryFlow):

    def __init__(self, U: float, V: float):
        super().__init__()
        self.u = lambda x,y: U
        self.v = lambda x,y: V

class Source(ElementaryFlow):

    def __init__(self, strength: float, position=(0, 0)):
        super().__init__()
        x0, y0 = position
        self.u = lambda x,y: strength * 2*(x-x0) / ((x-x0)**2 + (y-y0)**2)
        self.v = lambda x,y: strength * 2*(y-y0) / ((x-x0)**2 + (y-y0)**2)

class Doublet(ElementaryFlow):

    def __init__(self, strength: float, position=(0, 0)):
        super().__init__()


class FreeVortex(ElementaryFlow):

    def __init__(self, strength: float, position=(0, 0)):
        super().__init__()


class ComplexFlow:

    def __init__(self, flows: list[ElementaryFlow]):
        self.flows = flows
        self.u = lambda x,y: sum([flow.u(x,y) for flow in self.flows])
        self.v = lambda x,y: sum([flow.v(x,y) for flow in self.flows])

    def display(self, xlim, ylim, step):
        X = np.arange(-xlim, xlim, step)
        Y = np.arange(-ylim, ylim, step)

        x, y = np.meshgrid(X, Y)
        u = self.u(x,y)
        v = self.v(x,y)

        plt.figure(figsize=(10, 10))
        plt.streamplot(X, Y, u, v, density=1.5, linewidth=None, color="#2EA7C2")
        plt.title('Potential Flow')

        plt.grid()
        plt.show()


'''uni = Uniform(1, 0)
source = Source(1)
half_rankine = ComplexFlow([uni, source])
half_rankine.display(1, 1, 0.1)'''

uni1 = Uniform(1, 0)
source1 = Source(1, (-0.5, 0))
sink1 = Source(-1, (0.5, 0))
pair = ComplexFlow([uni1, source1, sink1])
pair.display(5, 5, 0.1)


'''# 1D arrays
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
plt.streamplot(X, Y, u, v, density=1.4, linewidth=None, color="#2EA7C2")
plt.plot(-1,0,'-or')
plt.plot(1,0,'-og')
plt.title('Potential Flow')

# Show plot with grid
plt.grid()
plt.show()'''