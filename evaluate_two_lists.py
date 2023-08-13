
import os

from gym_turret_defense_1 import TurretDefenseGym
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, save_leader_board, eval_list_of_models, update_elo
import random



if __name__ == "__main__":

    num_eval_games = 10

    #1 and 0 represent the team number
    active_team = 1
    passive_team = 0

    # Please pass a litst of passive agents you want to rank
    passive_nums = range(1,80)

    #Please pass a list of active agents you want to rank
    active_nums =  range(1,3)

    #Make a dict and add the models to samble to it , i may do this in a loop in the future.
    thread_list_elo_passive = {}
    thread_list_elo_active = {}

    for i in passive_nums:
             thread_list_elo_passive["Number_" + str(i) + "_team_" + str(passive_team)] = 1200 #TODO: Make fit in the general ELO update swag

    for i in active_nums:
             thread_list_elo_active["/Number_" + str(i) + "_team_" + str(active_team)] = 1200


    # Enviroment specifici parameters
    environment = TurretDefenseGym
    num_cpu = os.cpu_count()

    env = make_vec_env(environment, n_envs = num_cpu)

    model = PPO("MlpPolicy", env)

    #passive_model = model.load(passive_model, env=env)

    env.env_method("set_a", c1=1,
                   c2=.01,
                   passive_list=thread_list_elo_passive,
                   passive_model_type=model,
                   team=active_team,
                   terminal_state = 2,
                   single_mode_flag = False)



    # Create a vectorized enviroment to do parallel processing
    performance_dictionary = eval_list_of_models(thread_list_elo_active, env, model, num_eval_games,active_team)

    thread_list_elo_active = update_elo(performance_dictionary,thread_list_elo_active, active_team)

    save_leader_board(thread_list_elo_active)

    print(thread_list_elo_active)










