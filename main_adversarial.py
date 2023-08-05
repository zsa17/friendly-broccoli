import os

from gym_turret_adversarial_2 import TurretDefenseGym
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

    active = 1

    thread_list_elo = {}
    thread_list_elo["/Number_" + str(1520) + "_team_" + str(passive_team(active_team=str(active)))] = 1200

    single_mode_flag = False

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
                            team = active,
                            terminal_state = terminal_state,
                            single_mode_flag = single_mode_flag)

    model = PPO("MlpPolicy", env)

    print("starting the training")

    for i in range(100):

        # Start the learning processes
        model.learn(total_timesteps=1000000)

        model.save("Adversarial_team_" + str(active) + "iteration_" + str(i))

        print("One Done")










