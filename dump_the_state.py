import os

from gym_turret_defense_1 import TurretDefenseGym
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, save_leader_board, dump_state
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle



if __name__ == "__main__":

    #1 and 0 represent the team number
    num_eval = 500000

    active_team = 1
    passive_team = 0
    #pass_team_list = np.linspace(0, 113, num=114)
    active_num =  156

    with open('thread_list_elo.pickle', 'rb') as handle:
        thread_list_elo = pickle.load(handle)

    #print(thread_list_elo[str(passive_team)])

    # These are the specific stable baselines model that we are going to download per team
    active_model = "./model_directory/" + str(active_team) + "/Number_" + str(active_num) + "_team_" + str(active_team)
    #passive_model = "Adversarial_team_1iteration_1"

    #passive_model = "./model_directory/" + str(passive_team) + "/Number_" + str(nums) + "_team_" + str(passive_team)

    #Make a dict and add the models to samble to it , i may do this in a loop in the future.
    #thread_list_elo = {}

    #for nums in pass_team_list:
        #thread_list_elo["/Number_" + str(int(nums)) + "_team_" + str(passive_team)] = 1200

    # Enviroment specifici parameters
    environment = TurretDefenseGym
    num_cpu = os.cpu_count()

    env = make_vec_env(environment, n_envs = num_cpu)

    model = PPO("MlpPolicy", env)

    #passive_model = model.load(passive_model, env=env)

    env.env_method("set_a", c1=1,
                   c2=1,
                   passive_list=thread_list_elo[str(passive_team)],
                   passive_model_type=model,
                   team=active_team,
                   terminal_state = 2,
                   single_mode_flag = False)



    # Create a vectorized enviroment to do parallel processing


    dumped_state = dump_state([active_model], env, model, num_eval, active_team)

    df = pd.DataFrame(dumped_state)
    A = np.linspace(0, max(df[0]), num=100)
    B = np.linspace(min(df[1]), max(df[1]), num=100)

    print(min(B))
    print(max(B))

    C = pd.cut(df[0], A)
    D = pd.cut(df[1], B)
    combines = pd.concat([C, D])

    grouped = df.groupby(D)
    # loop through the rows using iterrows()
    state_dump_dict = {}
    matrix_index = {}
    for index, row in pd.DataFrame(grouped).iterrows():
        matrix_index[str(row[0])] = index
        state_dump_dict[str(row[0])] = pd.cut(row[1][0], A)

    grouped_C = df.groupby(C)
    matrix_index_C = {}
    for index, row in pd.DataFrame(grouped_C).iterrows():
        matrix_index_C[str(row[0])] = index

    dict_to_plat = {}
    list_ent = []
    for keys in state_dump_dict:
        list_ent = []
        for entries in state_dump_dict[keys]:
            list_ent += [matrix_index_C[str(entries)]]
        dict_to_plat[matrix_index[keys]] = list_ent

    matrix_value_array = np.zeros((100,100))

    for keys in dict_to_plat:
        for values in dict_to_plat[keys]:
            matrix_value_array[keys,values] += 1

    print(matrix_value_array)

    plt.matshow(matrix_value_array)
    plt.colorbar()
    plt.show()

    with open('matrix_value_array.pickle', 'wb') as handle:
        pickle.dump(matrix_value_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print("Done")










