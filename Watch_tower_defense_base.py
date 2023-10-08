import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Dynamics_stuff import IntegrateDynamics, TurretDynamics, IntegrateDynamics_own_velocity, IntegrateDynamics_own_xy
import random
import numpy as np
from utils import sample_from_dict_with_weight, passive_team, turret_controller

# Import collections module:
import collections


class TurretDefenseGymBase(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    """
    Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
    specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
    Example:
    """

    self.observation_space = spaces.Box(low=np.array([[0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,-1,-1,-1,-1]]),
                                        high=np.array([[1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1]]),
                                        dtype=np.float16)

    self.state = [0,0,0]
    self.action = [0,0]
    self.time_step = .1
    self.state_scale = [50, np.pi]
    self.c1 = 1
    self.c2 = 1
    self.passive_list = []
    self.passive_model = []
    self.passive_model_type = []
    self.single_mode_flag = False
    self.info = {"Test": "YOYO"}
    self.time_steps = 0
    self.terminal_state = 2
    # Initialize deque:
    self.state_send= collections.deque([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    self.velocity = 50
    self.turn_rate = .0001
    self.truncated = False
    self.teminated = False
    self.reaward = 0
    self.model_name = "temp"

    self.set_alpha = False
    self.set_alpha_value = 0



  def step(self, action):
    """
    This method is the primary interface between environment and agent.
    Paramters:
        action: int
                the index of the respective action (if action space is discrete)
    Returns:
        output: (array, float, bool)
                information provided by the environment about its current state:
                (observation, reward, done)
    """

    self.time_steps += self.time_step


    if self.team == 0:
      action = [self.passive_model.predict([self.state_send], deterministic = True )[0][0]*np.pi/6, action/6 - 1, self.velocity*(self.passive_model.predict([self.state_send], deterministic = True )[0][1]/6 - 1)]
      #action = [np.pi, action-1]

    elif self.team == 1:

      action = [action[0] * np.pi / 6, self.passive_model.predict([self.state_send], deterministic=True)[0] / 6 - 1, self.velocity*(action[1]/6-1)]


    self.state = IntegrateDynamics_own_velocity(self.state, self.time_step, action, self.velocity, self.turn_rate)

    if self.team == 0:
      self.reward = np.cos(self.state[1])
    elif self.team == 1:
      self.reward = -np.cos(self.state[1])

    self.info["action"] = action
    self.info["state"] = self.state
    self.info["time_step"] = self.time_steps


    self.terminated = False
    self.truncated = False


    if self.state[0] < self.terminal_state or self.state[0]>200:
      self.terminated = True
      if self.team == 0:
        self.reward += 5
      elif self.team == 1:
        self.reward -= 5

    if self.time_steps > 50:
      self.truncated = True

    self.state_transform = [self.state[0] / self.state_scale[0], np.cos(self.state[1]), np.sin(self.state[1]), np.cos(self.state[2]), np.sin(self.state[2])]

    for state in self.state_transform:
      self.state_send.append(state)
      self.state_send.popleft()

    #Incerase, decrease numebr of iteration per epoch.
    # Can we cycle through things lets dig deeper into the rsults.


    return [self.state_send], self.reward, self.terminated, self.truncated, self.info

  def set_a(self, c1, c2, passive_list, passive_model_type, team, terminal_state, single_mode_flag, set_alpha, set_alpha_value):
    self.c1 = c1
    self.c2 = c2
    self.passive_list = passive_list
    self.passive_model_type = passive_model_type
    self.team = team
    self.terminal_state = terminal_state
    self.single_mode_flag = single_mode_flag
    self.set_alpha = set_alpha
    self.set_alpha_value = set_alpha_value

  def reset(self,seed=None, options=None):
    """
    This method resets the environment to its initial values.
    Returns:
        observation:    array
                        the initial state of the environment
    """
    if self.single_mode_flag == False:
      self.model_name = list(sample_from_dict_with_weight(self.passive_list).keys())[0]
      self.passive_model = self.passive_model_type.load('model_directory/' + passive_team(str(self.team)) + '/' + self.model_name)

    self.time_steps = 0


    if self.set_alpha:
      self.state = [random.uniform(2, self.state_scale[0]), self.set_alpha_value, random.uniform(0, self.state_scale[1])]
    else:
      self.state = [random.uniform(2, self.state_scale[0]), random.uniform(0, self.state_scale[1]), random.uniform(0, self.state_scale[1])]




    self.state_transform = [self.state[0] , np.cos(self.state[1]), np.sin(self.state[1]), np.cos(self.state[2]), np.sin(self.state[2])]

    for state in self.state_transform :
      self.state_send.append(state)
      self.state_send.popleft()


    return [self.state_send], self.info

  def render(self, mode='human', close=False):
    """
    This methods provides the option to render the environment's behavior to a window
    which should be readable to the human eye if mode is set to 'human'.
    """
    pass

  def close(self):
    """
    This method provides the user with the option to perform any necessary cleanup.
    """
    pass
