import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##########################################################################################

##########################################################################################


class Planet:
    G = 6.67e-11

    def __init__(self, mass: float, position: tuple[float, float]):
        self.mass = mass
        self.x = position[0]
        self.y = position[1]

        self.vx: float
        self.vy: float
        self.ax: float 
        self.ay: float

        self.initial_defined = False

    def set_initial(self, vx: float, vy: float) -> None:
        self.vx = vx
        self.vy = vy
        self.initial_defined = True

    def get_distance_from(self, x: float, y: float) -> float:
        return math.sqrt((self.x-x)**2 + (self.y-y)**2)
    
    def get_force_from(self, planet) -> tuple[float, float]:
        r = self.get_distance_from(planet.x, planet.y)
        F = -self.G * self.mass * planet.mass / r**2
        Fx = F * (self.x - planet.x) / r
        Fy = F * (self.y - planet.y) / r
        return Fx, Fy
    

class System:

    def __init__(self, planets: np.ndarray[Planet]):
        self.planets = planets
        self.size = len(planets)

    def add_planet(self, planet: Planet) -> None:
        self.planets.append(planet)

    def del_planet(self, planet: Planet) -> None:
        self.planet = self.planet[self.planet == planet]
    
    def symplectic_evolve(self, dt: float, n_iter: int):

        position_output = np.zeros((self.size, n_iter), dtype=object)

        for k in range(n_iter):

            for n in range(self.size):
                planet = self.planets[n]
                Fx_total = 0
                Fy_total = 0

                if not planet.initial_defined:
                    raise Exception("Initial Conditions undefined, call Planet.set_initial(x,y)")
                
                for other_planet in self.planets:
                    if other_planet is planet:
                        continue
                    F = planet.get_force_from(other_planet)
                    Fx_total += F[0]
                    Fy_total += F[1]

                planet.ax = Fx_total / planet.mass
                planet.ay = Fy_total / planet.mass

                planet.vx = planet.vx + planet.ax*dt
                planet.vy = planet.vy + planet.ay*dt

                planet.x = planet.x + planet.vx*dt
                planet.y = planet.y + planet.vy*dt

                position_output[n, k] = (planet.x, planet.y)

        return position_output


class Visualizer:

    def __init__(self, position_output: np.ndarray):
        self.position_data = position_output
        self.num_planets = position_output.shape[0]
        print(position_output.shape)

    def set_axis(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        self.fig, self.ax = plt.subplots()
        self.ax.set(xlim=xlim, ylim=ylim)
        self.orbits = [self.ax.plot(*self.position_data[n, 0])[0] for n in range(self.num_planets)]  
        # n-th index is n-th planet's line plot

    def animate_orbit(self, dt_per_frame=100, dt_trail=1000):
        
        def update(frame):
            datapoints = self.position_data[:, max(0, dt_per_frame*frame-dt_trail) : dt_per_frame*frame]

            for n in range(self.num_planets):
                xdata = [coord[0] for coord in datapoints[n]]
                ydata = [coord[1] for coord in datapoints[n]]
                self.orbits[n].set_xdata(xdata)
                self.orbits[n].set_ydata(ydata)

            return self.orbits

        ani = animation.FuncAnimation(fig=self.fig, func=update, frames=10**6//dt_per_frame, interval=1)
        plt.show()
        
'''p1 = Planet(1, (-1,0))
p2 = Planet(1, (0,0))
p3 = Planet(1, (1,0))

p1.set_initial(0.18194288048334994,0.5148059977254023)
p2.set_initial(-0.3638857609666999,-1.0296119954508045)
p3.set_initial(0.18194288048334994,0.5148059977254023)'''

'''p1 = Planet(10, (-0.5,0))
p2 = Planet(10, (0.5,0))
p3 = Planet(10, (0.0207067154,0.3133550361))

p1.set_initial(0, 0)
p2.set_initial(0, 0)
p3.set_initial(0, 0)'''

p1 = Planet(100, (0.716248295713,0.384288553041))
p2 = Planet(100, (0.086172594591,1.342795868577))
p3 = Planet(100, (0.538777980808,0.481049882656))

p1.set_initial(1.245268230896,2.444311951777)
p2.set_initial(-0.67522432369,-0.96287961363)
p3.set_initial(-0.570043907206,-1.481432338147)

sys = System([p1, p2, p3])
pos = sys.symplectic_evolve(dt=0.001, n_iter=10**6)
print(pos)

vis = Visualizer(pos)
vis.set_axis((-1.5, 1.5), (-1.5, 1.5))
vis.animate_orbit()

