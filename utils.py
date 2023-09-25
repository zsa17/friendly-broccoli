import random
import os
import numpy as np
import copy
import json
import matplotlib.pyplot as plt

def passive_team(active_team):
    if active_team == "1":
        passive_team = "0"
    else:
        passive_team = "1"

    return passive_team

def plot_in_xy(d,alpha,beta, psi, omega, time):
    # some constants
    #va = 1
    #time_step = time[1] - time[0]
    d = np.multiply(d,20)
    alpha = np.multiply(alpha,np.pi)

    #First we need to calculate beta
    # beta = [np.divide(va,d[0])* np.sin(psi[0])*time_step]
    # for i in range(1,len(d)):
    #     beta += [beta[-1] + np.divide(va,d[i])* np.sin(psi[i])*time_step]

    # convert the set of equations to xy
    x_a = np.multiply(d,np.cos(beta))*time
    y_a = np.multiply(d,np.sin(beta))*time
    gamma = np.add(beta, alpha)

    return [x_a, y_a,gamma]


# Python 3 program for Elo Rating
import math


# Function to calculate the Probability


def Probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))


# Function to calculate Elo rating
# K is a constant.
# d determines whether
# Player A wins or Player B.
def EloRating(Ra, Rb, K, d):
    # To calculate the Winning
    # Probability of Player B
    Pb = Probability(Ra, Rb)

    # To calculate the Winning
    # Probability of Player A
    Pa = Probability(Rb, Ra)

    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if (d == 1):
        Ra = Ra + K * (1 - Pa)
        Rb = Rb + K * (0 - Pb)

    # Case -2 When Player B wins
    # Updating the Elo Ratings
    else:
        Ra = Ra + K * (0 - Pa)
        Rb = Rb + K * (1 - Pb)

    return Ra, Rb






def sample_from_dict_with_weight(d, sample=1):
    keys = random.choices(list(d), weights=(list(d.values())), k=sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))


import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = env_id()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def evaluate_model(models_to_compare, vec_env, model, num_cpu, num_evals):



    # num_cpu = os. cpu_count()  # Number of processes to use
    #if num_eval_models > 50 #TODO: Need to ad this check


    model_to_compare_dict = {models_to_compare[0]: 0, models_to_compare[1]: 0}
    obs = vec_env.reset()
    for indx, value in enumerate(models_to_compare):
        model.load(value)
        score = np.ones(num_cpu)
        indxs_list = list(range(0, num_cpu))
        for _ in range(num_evals//num_cpu):
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = vec_env.step(action)

                for check_indx, check_val in enumerate(indxs_list):
                    if dones[check_val] == True:
                        score[check_val] = rewards[check_val]
                        indxs_list.pop(check_indx)

                if sum(indxs_list) == 0:
                    model_to_compare_dict[value] = model_to_compare_dict[value] +  sum(score)
                    break


    return model_to_compare_dict, models_to_compare

def eval_list_of_models(active_models, vec_env, model, num_eval_games, active_team):



    # num_cpu = os. cpu_count()  # Number of processes to use
    #if num_eval_models > 50 #TODO: Need to ad this check
    evaluation_dictionary = {}
    for active_model in active_models:


        model.load("./model_directory/" + str(active_team)+ "/" + active_model)
        obs = vec_env.reset()

        obs_list_x = []
        obs_list_y = []
        beta = []
        psi = []
        omega = []
        time = []
        reward_list = []
        sum_end_d = 0
        evaluation_dictionary[active_model] = {"rewards":0, "num_games":0}

        reward_list = np.array(np.zeros(vec_env.num_envs))
        print("Evaluating" + active_model)
        while evaluation_dictionary[active_model]["num_games"] < num_eval_games:

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)

            reward_list += rewards
            indices = [i for i in range(len(dones)) if dones[i] == True]

            for indx in indices:
                evaluation_dictionary[active_model]["rewards"] += reward_list[indx]
                sum_end_d += obs[indx][0][0]
                evaluation_dictionary[active_model]["num_games"] += 1
                reward_list[indx] = 0

        evaluation_dictionary[active_model]["rewards"] = evaluation_dictionary[active_model]["rewards"]/evaluation_dictionary[active_model]["num_games"]
        average_end_d = sum_end_d/evaluation_dictionary[active_model]["num_games"]

        extra_info = [average_end_d]

    return evaluation_dictionary,extra_info


def eval_and_plot_model(active_model, vec_env, model, num_cpu):



    # num_cpu = os. cpu_count()  # Number of processes to use
    #if num_eval_models > 50 #TODO: Need to ad this check

    obs = vec_env.reset()
    model.load(active_model)
    score = np.ones(num_cpu)
    indxs_list = list(range(0, num_cpu))
    obs_list_x = []
    obs_list_y = []
    beta = []
    psi = []
    omega = []
    time = []
    reward_list = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0] == True:
            break
        obs_list_x += [info[0]["state"][0]]
        obs_list_y += [info[0]["state"][1]]
        beta += [info[0]["state"][2]]
        psi += [info[0]["action"][0]]
        omega += [info[0]["action"][1]]
        time += [info[0]["time_step"]]
        reward_list += [rewards[0]]







    xa,ya,gamma = plot_in_xy(obs_list_x, obs_list_y, beta,  psi, omega, time)


    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(obs_list_x, np.multiply(obs_list_y,180/np.pi), 'ob')
    axs[0, 0].set(xlabel='d', ylabel='alpha')
    axs[0, 1].plot(xa, ya, 'tab:orange')
    axs[1, 0].plot(time, reward_list, 'tab:green')
    axs[1, 0].set(xlabel='time', ylabel='reward')
    axs[1, 1].plot(time, omega, 'tab:red')
    axs[1, 1].set(xlabel='time', ylabel='omega control')
    axs[2, 1].plot(time, np.multiply(psi,(180/np.pi)), 'tab:red')
    axs[2, 1].set(xlabel='time', ylabel='psi control')
    axs[2, 0].plot(time, gamma*180/np.pi, 'tab:red')
    axs[2, 0].set(xlabel='time', ylabel='look angle')

    plt.show()




def should_we_move_on(score_1, score_2, desired_percent):
        if (score_1 - score_2)/score_1 > desired_percent:
            return True
        else:
            return False


def save_leader_board(details):
    with open('LeaderBoard.txt', 'w') as leader_board:
        leader_board.write(json.dumps(details, indent=2))

def save_reward_board(details):
    with open('AverageReward.txt', 'w') as leader_board:
        leader_board.write(json.dumps(details, indent=2))

def save_extra_board(details):
    with open('extra_data.txt', 'w') as leader_board:
        leader_board.write(json.dumps(details, indent=2))

def update_elo(performation_dictionary, elo_dictionary, team):
    for keys in performation_dictionary:
        for keys_2 in performation_dictionary:

            Ra = elo_dictionary[keys]
            Rb = elo_dictionary[keys_2]
            K = 30

            if performation_dictionary[keys]["rewards"] > performation_dictionary[keys_2]["rewards"]:
                d = 1
                Ra,Rb =EloRating(Ra, Rb, K, d)
                elo_dictionary[keys] = Ra
                elo_dictionary[keys_2] = Rb
            elif performation_dictionary[keys]["rewards"] < performation_dictionary[keys_2]["rewards"]:
                d = 2
                EloRating(Ra, Rb, K, d)
                Ra = elo_dictionary[keys]
                Rb = elo_dictionary[keys_2]
                elo_dictionary[keys]= Ra
                elo_dictionary[keys_2] = Rb
            else:
                pass

    return elo_dictionary

def turret_controller(alpha):
    if alpha>0:
        action = -1
    elif alpha<0:
        action = 1
    else:
        action = 0

    return action

def attack_controller(alpha):
    if alpha>0:
        action = np.pi- .1
    elif alpha<0:
        action = np.pi + .1
    else:
        action = 0

    return action

def start_training():
    import os

    dir_path = r'model_directory/1'

    number = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

    if number == 0:
        return 1

    else:
        return(len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])-1)


#def scores_to_elo(player_1_elo, player_2_elo, player_1_score, player_2_score, max_min_flag):



