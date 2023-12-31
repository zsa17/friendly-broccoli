import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Dynamics_stuff import IntegrateDynamics, TurretDynamics, IntegrateDynamics_own, IntegrateDynamics_own_xy
import random
import numpy as np
from utils import sample_from_dict_with_weight, passive_team

class TurretDefenseGym(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    """
    Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
    specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
    Example:
    """
    self.observation_shape = (1, 2)
    self.observation_space = spaces.Box(low=np.array([[-1,-1]]),
                                        high=np.array([[1, 1]]),
                                        dtype=np.float16)


    self.action_space = spaces.Discrete(3, start=-1, seed=42)
    self.state = [0,0]
    self.action = [0,0]
    self.time_step = .1
    self.state_scale = [20, 2*np.pi]
    self.team ="RED"
    self.c1 = 1
    self.c2 = 1
    self.passive_list = []
    self.passive_model = []
    self.passive_model_type = []
    self.single_mode_flag = False
    self.info = {"Test": "YOYO"}
    self.time_steps = 0
    self.terminal_state = 2


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
      #action = [self.passive_model.predict([self.state])[0][0][0], np.sign(action[0][0])]
      action = [0, action]

    elif self.team == 1:
      #action = [action[0][0], np.sign(self.passive_model.predict([self.state])[0][0][0])]
      action = [action[0][0]*np.pi, np.pi]


    #self.state = IntegrateDynamics(TurretDynamics,self.state,self.time_step, action)[-1]
    self.state = [self.state[0] * self.state_scale[0],self.state[1] * self.state_scale[1]]


    reward = self.state[1]


    #self.state = IntegrateDynamics_own(self.state, self.time_step, action)
    self.state = [self.state[0], self.state[1] + action[1]]




    self.info["action"] = action
    self.info["time_step"] = self.time_steps

    terminated = False
    truncated = False



    if self.time_steps > 100:
      terminated = True


    self.state = [self.state[0] / self.state_scale[0],self.state[1] / self.state_scale[1]]



    return self.state, reward, terminated, truncated, self.info

  def set_a(self, c1, c2, passive_list, passive_model_type, team, terminal_state, single_mode_flag):
    self.c1 = c1
    self.c2 = c2
    self.passive_list = passive_list
    self.passive_model_type = passive_model_type
    self.team = team
    self.terminal_state = terminal_state
    self.single_mode_flag = single_mode_flag

  def reset(self,seed=None, options=None):
    """
    This method resets the environment to its initial values.
    Returns:
        observation:    array
                        the initial state of the environment
    """
    if self.single_mode_flag == False:
      self.passive_model = self.passive_model_type.load('model_directory/' + passive_team(str(self.team)) + '/' + list(sample_from_dict_with_weight(self.passive_list).keys())[0])

    self.time_steps = 0
    self.state = [random.uniform(.33, 1), random.uniform(0, 1)]
    #self.state = [50,0]

    return self.state, self.info

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
