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

def plot_in_xy(d,alpha, psi, omega, time):
    # some constants
    va = 1

    #First we need to calculate beta
    beta = np.divide(va,d)* np.sin(psi)*time

    # convert the set of equations to xy
    x_a = np.multiply(d,np.cos(beta))*time
    y_a = np.multiply(d,np.sin(beta))*time
    gamma = np.add(beta, alpha)*time

    plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.plot(x_a, y_a, 'ro')

    plt.show()





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

def evaluate_model(thread_list_master, vec_env, team, model, num_eval_models, num, num_cpu, num_evals):



    # num_cpu = os. cpu_count()  # Number of processes to use
    #if num_eval_models > 50 #TODO: Need to ad this check

    models_to_compare =[ "./model_directory/" + str(team) + "/Number_" + str(num) + "_team_" + str(team),"./model_directory/" + str(team) + "/Number_" + str(num-1) + "_team_" + str(team)]

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
                        indxs_list[check_indx] = 0

                if sum(indxs_list) == 0:
                    thread_list_master[str(team)][models_to_compare[indx].split("/")[-1]]["score"] = thread_list_master[str(team)][models_to_compare[indx].split("/")[-1]]["score"] +  sum(score)
                    break


    return thread_list_master, models_to_compare

def eval_and_plot_model(active_model, vec_env, model, num_cpu):



    # num_cpu = os. cpu_count()  # Number of processes to use
    #if num_eval_models > 50 #TODO: Need to ad this check

    obs = vec_env.reset()
    model.load(active_model)
    score = np.ones(num_cpu)
    indxs_list = list(range(0, num_cpu))
    obs_list_x = []
    obs_list_y = []
    psi = []
    omega = []
    time = []
    reward_list = []
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0] == True:
            break
        obs_list_x += [obs[0][0][0]]
        obs_list_y += [obs[0][0][1]]
        psi += [info[0]["action"][0]]
        omega += [info[0]["action"][1]]
        time += [info[0]["time_step"]]
        reward_list += [rewards[0]]






    plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.plot(obs_list_x, obs_list_y, 'bo')

    plt.show()

    plot_in_xy(obs_list_x, obs_list_y, psi, omega, time)

    plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.plot(time, reward_list, 'go')

    plt.show()




def should_we_move_on(score_1, score_2, desired_percent):
        if (score_1 - score_2)/score_1 > desired_percent:
            return True
        else:
            return False


def save_leader_board(details):
    with open('LeaderBoard.txt', 'w') as leader_board:
        leader_board.write(json.dumps(details, indent=2))


#def scores_to_elo(player_1_elo, player_2_elo, player_1_score, player_2_score, max_min_flag):



