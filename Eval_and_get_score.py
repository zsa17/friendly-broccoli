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
        model = PPO("MlpPolicy", env_dict[str(i)], n_epochs=1)
        model.save("./model_directory/" + str(i) + "/Number_0_team_" + str(i))
        model.save("./model_directory/" + str(i) + "/Number_1_team_" + str(i))


    num = 2 #Start the counter




    print("Starting New Round")
    passive_model = 'Number_24_team_1'
    team = 0

    # Set the enviroment specific parameters and send it in models type and list.
    env_dict[str(team)].env_method("set_a", c1 = .1,
                            c2 = 1,
                            #passive_list = thread_list_elo[passive_team(str(team))],
                            passive_list= {passive_model: 1200},
                            passive_model_type = model,
                            team = team,
                            terminal_state = terminal_state,
                            single_mode_flag = False)

    # Load the model weights that are about to be trained
    model = PPO("MlpPolicy", env_dict[str(team)], n_epochs=1)
    models_to_compare = ["./model_directory/" + str(team) + "/Number_" + str(23) + "_team_" + str(team),
                         "./model_directory/" + str(team) + "/Number_" + str(25) + "_team_" + str(team)]

    # Evaluate the trained model
    result_end, models_to_compare = evaluate_model(models_to_compare, env_dict[str(team)], model, num_cpu,num_evals)

    print(result_end)





