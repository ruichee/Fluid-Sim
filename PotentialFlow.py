import numpy as np
import matplotlib.pyplot as plt

#####################################################################################

class ElementaryFlow:
    def __init__(self, position):
        self.u: function[float, float]
        self.v: function[float, float]
        self.x0, self.y0 = position
        self.r = lambda x,y: ((x-self.x0)**2 + (y-self.y0)**2)**(1/2)
        self.pos_color = None

class Uniform(ElementaryFlow):
    def __init__(self, U: float, V: float):
        super().__init__(position=(0, 0))
        self.u = lambda x,y: U
        self.v = lambda x,y: V

class Source(ElementaryFlow):
    def __init__(self, strength: float, position=(0, 0)):
        super().__init__(position)
        self.u = lambda x,y: strength * 2*(x-self.x0) / self.r(x,y)**2
        self.v = lambda x,y: strength * 2*(y-self.y0) / self.r(x,y)**2
        self.pos_color = "green" if strength > 0 else "red"

class Doublet(ElementaryFlow):
    def __init__(self, strength: float, position=(0, 0)):
        super().__init__(position)
        self.u = lambda x,y: strength * ((y-self.y0)**2 - (x-self.x0)**2) / self.r(x,y)**4
        self.v = lambda x,y: strength * (-2*(y-self.y0)*(x-self.x0)) / self.r(x,y)**4
        self.pos_color = "blue"

class FreeVortex(ElementaryFlow):
    def __init__(self, strength: float, position=(0, 0)):
        super().__init__(position)
        self.u = lambda x,y: strength * -((y-self.y0)/(x-self.x0)**2) / (1 + ((y-self.y0)/(x-self.x0))**2)
        self.v = lambda x,y: strength * (1/(x-self.x0)) / (1 + ((y-self.y0)/(x-self.x0))**2)
        self.pos_color = "purple"

#####################################################################################

class ComplexFlow:

    def __init__(self, flows: list[ElementaryFlow]):
        self.flows = flows
        self.u = lambda x,y: sum([flow.u(x,y) for flow in self.flows])
        self.v = lambda x,y: sum([flow.v(x,y) for flow in self.flows])

    def display(self, xlim, ylim, step=0.01, density=2.0):
        X = np.arange(-xlim, xlim+step, step)
        Y = np.arange(-ylim, ylim+step, step)

        x, y = np.meshgrid(X, Y)
        u = self.u(x,y)
        v = self.v(x,y)

        plt.figure(figsize=(10, 10))
        plt.streamplot(x, y, u, v, density=density, broken_streamlines=True, \
                       linewidth=None, start_points=None, color="#2EA7C2")
        
        # use of non-broken streamlines is inefficient and produces uneven streamlines
        # use of start_points allows better representation of streamline density but 
        # which start points to choose depends on the scenario, can consider adding feature to automate
        # for uniform flow: start_pts = [(-xlim, y) for y in np.linspace(-ylim, ylim-step, 30)]
        # for source/sink: point slight outside the source/sink point in all directions
        # broken streamlines is chosen, represents shape well, but not real streamline density
        
        for flow in self.flows:
            if flow.pos_color:
                plt.scatter(x=flow.x0, y=flow.y0, marker='o', color=flow.pos_color)
        
        plt.title('Potential Flow')
        plt.xlim((-xlim, xlim))
        plt.ylim((-ylim, ylim))
        plt.grid()
        plt.show()

#####################################################################################

# half bluff body
uni = Uniform(1, 0)
source = Source(1)
half_rankine = ComplexFlow([uni, source])
half_rankine.display(5, 5, 0.01)

# full bluff body
source1 = Source(1, (-0.5, 0))
sink1 = Source(-1, (0.5, 0))
pair = ComplexFlow([uni, source1, sink1])
pair.display(5, 5, 0.01)

# flow around cylinder
doub = Doublet(1)
cyl = ComplexFlow([uni, doub])
cyl.display(3, 3, 0.01)

# flow around rotating cylinder
vortex = FreeVortex(1)
rotcyl = ComplexFlow([uni, doub, vortex])
rotcyl.display(3, 3, 0.01)

# drain
sink = Source(-1)
vort = FreeVortex(3)
drain = ComplexFlow([sink, vort])
drain.display(3, 3, 0.01)