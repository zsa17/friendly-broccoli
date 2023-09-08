import os

from gym_turret_defense_0 import TurretDefenseGym as TurretDefenseGym_0
from gym_turret_defense_1 import TurretDefenseGym as TurretDefenseGym_1
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, save_leader_board, eval_list_of_models,update_elo,save_reward_board, save_extra_board, turret_controller
import random
import numpy as np
from gymnasium import spaces
import pickle
import torch as th






if __name__ == "__main__":


    # Specific parameters for the self-play system
    how_many_teams_do_you_have = 2
    number_tournaments = 5000
    num_eval_models = 20
    move_on_threshold = 50
    num_evals = 20
    terminal_state = 2
    train_for_time_steps = 10000

    # Enviroment specifici parameters
    environment = {"0":TurretDefenseGym_0, "1": TurretDefenseGym_1}
    num_cpu = os.cpu_count()

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))


    # Initialize you infomraiton dictionaroes
    thread_list_elo = {}
    performance_dict = {}
    thread_list_master = {}
    counter = {}
    env_dict = {}
    extra_data = {}
    # Loop through number of teams and start the initial dictionary structure.
    for i in range(how_many_teams_do_you_have):

        thread_list_elo[str(i)] = {"Number_0_team_" + str(i) : 1200, "Number_1_team_" + str(i) : 1200}
        performance_dict[str(i)] = {"Number_0_team_" + str(i): {"Round_1": 0 }, "Number_1_team_" + str(i): {"Round_1": 0 }}
        extra_data[str(i)] = {"Number_0_team_" + str(i): [0,0], "Number_1_team_" + str(i): [0,0]}

        # Create a vectorized enviroment to do parallel processing
        env_dict[str(i)] = make_vec_env(environment[str(i)], n_envs=num_cpu)

        model = PPO("MlpPolicy", env_dict[str(i)], n_epochs=2, policy_kwargs=policy_kwargs)


        counter[str(i)] = 985



    # If you want to pre-populate
    # for i in range(2):
    #     for j in range(how_many_teams_do_you_have):
    #
    #          thread_list_elo[str(j)]["Number_" + str(i) + "_team_" + str(j)] = 1200

    if os.path.isfile('thread_list_elo.pickle') and os.path.isfile('performance_dict.pickle'):

        with open('performance_dict.pickle', 'rb') as handle:
            performance_dict = pickle.load(handle)

        with open('thread_list_elo.pickle', 'rb') as handle:
            thread_list_elo = pickle.load(handle)

        with open('extra_data.pickle', 'rb') as handle:
            extra_data = pickle.load(handle)

    num = 1#Start the counter

    while num < number_tournaments:
        for team in range(how_many_teams_do_you_have):
            print("Starting New Round")
            move_on_check = True
            train_again_flag = True
            time_step_trained = 0
            while move_on_check:


                # Set the enviroment specific parameters and send it in models type and list.
                env_dict[str(team)].env_method("set_a", c1 = 1,
                                        c2 = .01,
                                        #passive_list = thread_list_elo[passive_team(str(team))],
                                        passive_list= thread_list_elo[passive_team(str(team))],
                                        passive_model_type = model,
                                        team = team,
                                        terminal_state = terminal_state,
                                        single_mode_flag = False)

                # Load the model weights that are about to be trained
                model = PPO("MlpPolicy", env_dict[str(team)], n_epochs=2, policy_kwargs=policy_kwargs)

                performance_dictionary, extra_info = eval_list_of_models(thread_list_elo[str(team)], env_dict[str(team)], model,
                                                             num_evals, team)



                thread_list_elo_active = update_elo(performance_dictionary, thread_list_elo[str(team)], team)
                thread_list_elo[str(team)] = thread_list_elo_active

                move_on_check = True
                train_again_flag = False

                save_leader_board(thread_list_elo)

                with open('thread_list_elo.pickle', 'wb') as handle:
                    pickle.dump(thread_list_elo, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('performance_dict.pickle', 'wb') as handle:
                    pickle.dump(performance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('extra_data.pickle', 'wb') as handle:
                    pickle.dump(extra_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        num += 1







