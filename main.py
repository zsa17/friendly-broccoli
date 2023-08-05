import os

from gym_turret_defense_0 import TurretDefenseGym as TurretDefenseGym_0
from gym_turret_defense_1 import TurretDefenseGym as TurretDefenseGym_1
from stable_baselines3 import PPO
from population import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import passive_team, evaluate_model, make_env, should_we_move_on, save_leader_board
import random
import numpy as np
from gymnasium import spaces






if __name__ == "__main__":


    # Specific parameters for the self-play system
    how_many_teams_do_you_have = 2
    number_tournaments = 5000
    num_eval_models = 50
    move_on_threshold = 50
    num_evals = 50
    terminal_state = 2

    # Enviroment specifici parameters
    environment = {"0":TurretDefenseGym_0, "1": TurretDefenseGym_1}
    num_cpu = os.cpu_count()


    # Initialize you infomraiton dictionaroes
    thread_list_elo = {}
    thread_list_master = {}
    counter = {}
    action_space = {"0": spaces.Discrete(3, start=-1, seed=42), "1": spaces.Box(low=-1.0, high=1.0, shape=(1, 1), dtype=np.float32)}
    env_dict = {}

    # Loop through number of teams and start the initial dictionary structure.
    for i in range(how_many_teams_do_you_have):

        thread_list_elo[str(i)] = {"Number_0_team_" + str(i) : 1200, "Number_1_team_" + str(i) : 1200}
        thread_list_master[str(i)] = {"Number_0_team_" + str(i): {"score": 0, "elo": 1200}, "Number_1_team_" + str(i): {"score": 0, "elo": 1200}}

        # Create a vectorized enviroment to do parallel processing
        env_dict[str(i)] = make_vec_env(environment[str(i)], n_envs=num_cpu)
        # Create and define the model that you will be using.
        model = PPO("MlpPolicy", env_dict[str(i)])
        model.save("./model_directory/" + str(i) + "/Number_0_team_" + str(i))
        model.save("./model_directory/" + str(i) + "/Number_1_team_" + str(i))

        counter[str(i)] = 1524
        team_name = [0,1]
    for i in range(1525):
        for j in team_name:
             thread_list_elo[str(j)]["Number_" + str(i) + "_team_" + str(j)] = 1200
             thread_list_master[str(j)]["Number_" + str(i) + "_team_" + str(j)] = {"score": 0, "elo": 1200}



    num = 1524 #Start the counter


    while num < number_tournaments:
        for team in range(how_many_teams_do_you_have):

            print("Starting New Round")

            # Set the enviroment specific parameters and send it in models type and list.
            env_dict[str(team)].env_method("set_a", c1 = .1,
                                    c2 = 1,
                                    #passive_list = thread_list_elo[passive_team(str(team))],
                                    passive_list= {list(thread_list_elo[passive_team(str(team))])[-1]: 1200},
                                    passive_model_type = model,
                                    team = team,
                                    terminal_state = terminal_state,
                                    single_mode_flag = False)

            # Load the model weights that are about to be trained
            model = PPO("MlpPolicy", env_dict[str(team)])
            model.load("./model_directory/" + str(team) + "/Number_" + str(counter[str(team)]) + "_team_" + str(team), env=env_dict[str(team)])

            # Start the learning processes
            model.learn(total_timesteps=10000000)

            # Evaluate the trained model
            thread_list_master, models_to_compare = evaluate_model(thread_list_master, env_dict[str(team)], team, model, num_eval_models,counter[str(team)] , num_cpu,num_evals)

            # Did the model sufficiently improve?
            move_on_check = should_we_move_on(thread_list_master[str(team)][models_to_compare[0].split("/")[-1]]["score"],
                                              thread_list_master[str(team)][models_to_compare[1].split("/")[-1]]["score"],
                                              move_on_threshold)

            # Short ciruit to make the model move on all the time anyway
            move_on_check = True

            if move_on_check:
                counter[str(team)] += 1
                model.save("./model_directory/" + str(team) + "/Number_" + str(counter[str(team)]) + "_team_" + str(team))
                thread_list_elo[str(team)]["Number_" + str(counter[str(team)] ) + "_team_" + str(team)] = 1200
                thread_list_master[str(team)]["Number_" + str(counter[str(team)] ) + "_team_" + str(team)] = {"score": 0, "elo": 1200}
                save_leader_board(thread_list_elo)

                if terminal_state > 2:
                    terminal_state = terminal_state - 1

            else:
                continue

        num += 1







