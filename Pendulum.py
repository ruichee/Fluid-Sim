import numpy as np
import matplotlib.pyplot as plt

# for single pendulum with non-small angle 
# F(tangential) = ma 
# -mgsinθ - cω = mrα, where r = L
# d2θ/dt2 + dθ/dt + gsinθ/L = 0

# ode derived from first principles
ode = lambda theta, omega: -g*np.sin(theta)/L - c*omega

# parameters
dt = 0.01           # time step
g = 9.81            # gravitational acceleration
L = 1               # bar length
c = 0.1             # damping constant
n_iter = 3000       # iterations

# initial conditions
theta = np.pi / 2 * 1.8
omega = 0
alpha = ode(theta, omega)

# data storage
theta_lst = [theta*180/np.pi]
omega_lst = [omega]
alpha_lst = [alpha]

# semi implicit euler scheme (symplectic)
for i in range(n_iter):

    omega = omega + alpha*dt
    theta = theta + omega*dt
    alpha = ode(theta, omega)

    theta_lst.append(theta*180/np.pi)
    omega_lst.append(omega)
    alpha_lst.append(alpha)

# plot resulting quantities
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(np.linspace(0, n_iter*dt, n_iter+1), theta_lst)
ax2.plot(np.linspace(0, n_iter*dt, n_iter+1), omega_lst)
ax3.plot(np.linspace(0, n_iter*dt, n_iter+1), alpha_lst)
plt.show()

