import numpy as np
from scipy.integrate import RK45
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def TurretDynamics(state, t):
  d = state[0]
  alpha = state[1]

  # set the constants
  psi = 1
  omega = 1
  v = 1

  dd = v * np.cos(psi)
  alphad = omega - (v/d)*np.sin(psi)



  return [dd, alphad]

def IntegrateDynamics(Dynamics, starting_state,time_step, action):

  t = arange(0.0, 1.0, .1)
  Dynamics.psi = action[0]
  Dynamics.omega = action[1]
  #state = odeint(Dynamics, starting_state, t)[-1]
  state = RK45(Dynamics, 0, starting_state, 1, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False, first_step=None)


  return (state)


def IntegrateDynamics_own(state,time_step, action, velocity, turn_rate):
  d = state[0]
  alpha = state[1]
  beta = state[2]

  # set the constants
  psi = action[0]
  omega = action[1]
  v = velocity
  turn_rate = turn_rate

  dd = v * np.cos(psi)
  alphad = turn_rate*omega - (v/d)*np.sin(psi)
  betad = (v/d)*np.sin(psi)

  d = dd*time_step + d
  alpha = alphad*time_step + alpha
  beta = betad*time_step + beta

  state = [d, alpha, beta]

  return (state)

def IntegrateDynamics_own_EP(state, time_step, action, velocity_pursuer, velocity_evader,min_evader_radi,min_pursuer_radi, theta):
    x = state[0]
    y = state[1]

    # set the constants
    psi = action[1]
    omega = action[0]

    # Constants
    vp = velocity_pursuer
    ve = velocity_evader
    Re = min_evader_radi
    Rp = min_pursuer_radi

    dx = vp * np.sin(theta) - (ve*y*omega)/Re
    dy = vp*np.cos(theta) - ve + (ve*x*omega)/Re
    dtheta = -(ve*omega)/Re + (vp*psi)/Rp

    x = dx * time_step + x
    y = dy * time_step + y
    theta = dtheta * time_step + theta

    state = [x, y, theta, theta]

    print("From dynamics" + str(state))

    return (state)

def IntegrateDynamics_own_xy(state,time_step, action):
  x = state[0]
  y = state[1]
  gamma = state[2]

  # set the constants
  psi = action[0]
  omega = action[1]
  v = 1
  turn_rate = 10

  xd = v*np.cos(psi)
  yd = v*np.sin(psi)
  gammad = turn_rate*omega

  x = xd*time_step + x
  y = yd*time_step + y
  gamma = gammad*time_step + gamma


  state = [x,y,gamma]




  return (state)




