import os

from gym_turret_defense_1 import TurretDefenseGym
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, save_leader_board
import random



if __name__ == "__main__":


    # Specific parameters for the self-play system
    how_many_teams_do_you_have = 2
    number_tournaments = 50
    num_eval_models = 50
    move_on_threshold = 50
    num_evals = 50
    terminal_state = 2

    thread_list_elo = {}
    thread_list_elo["/Number_" + str(1520) + "_team_" + str(passive_team)] = 1200

    single_mode_flag = True

    # Enviroment specifici parameters
    environment = TurretDefenseGym
    num_cpu = os.cpu_count()


    # Create a vectorized enviroment to do parallel processing
    env = make_vec_env(environment, n_envs = num_cpu)

    # Create and define the model that you will be using.
    model = PPO("MlpPolicy", env)


    # Set the enviroment specific parameters and send it in models type and list.
    env.env_method("set_a", c1 = .1,
                            c2 = 1,
                            passive_list = thread_list_elo,
                            passive_model_type = model,
                            team = 1,
                            terminal_state = terminal_state,
                            single_mode_flag = single_mode_flag)


    print("starting the training")

    # Start the learning processes
    model = PPO("MlpPolicy", env).learn(total_timesteps=100000)

    model.save("Test_Agent")









