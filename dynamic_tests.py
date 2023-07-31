import numpy as np

from Dynamics_stuff import IntegrateDynamics, TurretDynamics, IntegrateDynamics_own
import matplotlib.pyplot as plt



# This is a test to make sure dynamics are working
state = [100, 0 ]
obs_list_x = []
obs_list_y = []
reward = []
for i in range(100):


    control = [np.pi, 0] #(psi, omega)
    state = IntegrateDynamics_own(state,1, control)
    reward += [.5 * (1 + np.cos(state[1])) + 1]
    obs_list_x += [state[0]]
    obs_list_y += [state[1]]

plt.style.use('_mpl-gallery')

# plot
fig, ax = plt.subplots()

ax.plot(reward, obs_list_y, 'bo')

plt.show()


