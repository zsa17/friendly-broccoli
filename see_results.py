import os

from gym_turret_defense_1 import TurretDefenseGym
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, save_leader_board, eval_and_plot_model
import random



if __name__ == "__main__":

    #1 and 0 represent the team number
    active_team = 1
    passive_team = 0
    passive_num = 53
    active_num =  52

    # These are the specific stable baselines model that we are going to download per team
    active_model = "./model_directory/" + str(active_team) + "/Number_" + str(active_num) + "_team_" + str(active_team)
    #passive_model = "Adversarial_team_1iteration_1"

    passive_model = "./model_directory/" + str(passive_team) + "/Number_" + str(passive_num) + "_team_" + str(passive_team)

    #Make a dict and add the models to samble to it , i may do this in a loop in the future.
    thread_list_elo = {}
    thread_list_elo["/Number_" + str(passive_num) + "_team_" + str(passive_team)] = 1200

    # Enviroment specifici parameters
    environment = TurretDefenseGym
    num_cpu = os.cpu_count()

    env = make_vec_env(environment, n_envs = num_cpu)

    model = PPO("MlpPolicy", env)

    #passive_model = model.load(passive_model, env=env)

    env.env_method("set_a", c1=1,
                   c2=1,
                   passive_list=thread_list_elo,
                   passive_model_type=model,
                   team=active_team,
                   terminal_state = 2,
                   single_mode_flag = False)



    # Create a vectorized enviroment to do parallel processing


    eval_and_plot_model(active_model, env, model, num_cpu)










