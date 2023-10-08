import os

from gym_turret_defense_1 import TurretDefenseGym
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, eval_list_of_models, dump_state
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch as th



if __name__ == "__main__":

    #1 and 0 represent the team number
    num_eval = 5000

    active_team = 1
    passive_team = 0
    #pass_team_list = np.linspace(0, 113, num=114)
    active_num =  78
    Performance_list = []
    passive_num_base = 78

    singular_surface_list = [np.pi, 0]
    for passive_iterator in range(78):
        passive_num = passive_num_base - passive_iterator
        print("Attacking Model Team " + str(passive_team) + " Number " + str(passive_num) )
        for alpha_value in singular_surface_list:
            passive_model = "./model_directory/" + str(passive_team) + "/Number_" + str(passive_num) + "_team_" + str(
                passive_team)
            # Make a dict and add the models to samble to it , i may do this in a loop in the future.
            thread_list_elo = {}
            thread_list_elo["/Number_" + str(passive_num) + "_team_" + str(passive_team)] = 1200
            #print(thread_list_elo[str(passive_team)])

            # These are the specific stable baselines model that we are going to download per team
            active_model = "/Number_" + str(active_num) + "_team_" + str(active_team)
            baseline_model = "/Number_" + str(active_num-1) + "_team_" + str(active_team)
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

            # Design the Neural Network that will be used
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                                 net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))

            # Create and define the model that you will be using.
            model = PPO("MlpPolicy", env, n_epochs=10, policy_kwargs=policy_kwargs)

            #passive_model = model.load(passive_model, env=env)

            env.env_method("set_a", c1=1,
                           c2=1,
                           passive_list=thread_list_elo,
                           passive_model_type=model,
                           team=active_team,
                           terminal_state = 2,
                           single_mode_flag = False,
                           set_alpha = True,
                           set_alpha_value = alpha_value)

            # Start the learning processes
            print("Started Adversarial Training " + str(alpha_value))

            model.learn(total_timesteps=300000000)

            print("Completed Training " + str(alpha_value))

            model.save("model_directory/1/Adversarail_" + str(active_team) + "_" + str(alpha_value))
            adversarial_model = "Adversarail_" + str(active_team) + "_" + str(alpha_value)









            # Create a vectorized enviroment to do parallel processing

            performance_dictionary, extra_info = eval_list_of_models([active_model,baseline_model,adversarial_model], env, model,
                                                                     num_eval, active_team)

            Performance_list += [performance_dictionary]

            with open('Performance_list.pickle', 'wb') as handle:
                pickle.dump(Performance_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("Completed Evaluation " + str(alpha_value))

            print(performance_dictionary)
            print(extra_info)






