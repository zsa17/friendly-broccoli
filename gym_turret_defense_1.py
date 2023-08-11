import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Dynamics_stuff import IntegrateDynamics, TurretDynamics, IntegrateDynamics_own, IntegrateDynamics_own_xy
import random
import numpy as np
from utils import sample_from_dict_with_weight, passive_team
from gym_turret_defense_base import TurretDefenseGymBase

class TurretDefenseGym(TurretDefenseGymBase):
  metadata = {'render.modes': ['human']}

  def __init__(self):

    """
    Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
    specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
    Example:
    """

    # self.action_space = spaces.Box(low=0, high=2, shape=(1, 1), dtype=np.float32)
    self.action_space = spaces.Discrete(12, start=0, seed=42)

    super().__init__()
