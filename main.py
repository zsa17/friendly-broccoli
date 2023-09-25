import os
from gym_turret_defense_0 import TurretDefenseGym as TurretDefenseGym_0
from gym_turret_defense_1 import TurretDefenseGym as TurretDefenseGym_1
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from utils import passive_team, save_leader_board, eval_list_of_models,update_elo,save_reward_board, save_extra_board, start_training
import pickle
import torch as th






if __name__ == "__main__":


    # Specific parameters for the self-play system
    how_many_teams_do_you_have = 2
    number_tournaments = 5000
    move_on_threshold = 1.55
    num_evals = 50
    terminal_state = 2
    train_for_time_steps = 10000
    num_cpu = os.cpu_count()

    # Dictionary of Enviroments
    environment = {"0":TurretDefenseGym_0, "1": TurretDefenseGym_1}


    # Design the Neural Network that will be used
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))


    # Initialize you infomraiton dictionaroes
    thread_list_elo = {}
    performance_dict = {}
    counter = {}
    env_dict = {}
    extra_data = {}

    # Loop through number of teams and start the initial dictionary structure.
    for i in range(how_many_teams_do_you_have):

        # Make the dictionaries we will use to write JSON data
        thread_list_elo[str(i)] = {"Number_0_team_" + str(i) : 1200, "Number_1_team_" + str(i) : 1200}
        performance_dict[str(i)] = {"Number_0_team_" + str(i): {"Round_1": 0 }, "Number_1_team_" + str(i): {"Round_1": 0 }}
        extra_data[str(i)] = {"Number_0_team_" + str(i): [0,0], "Number_1_team_" + str(i): [0,0]}

        # Create a vectorized enviroment to do parallel processing
        env_dict[str(i)] = make_vec_env(environment[str(i)], n_envs=num_cpu)

        # Create and define the model that you will be using.
        model = PPO("MlpPolicy", env_dict[str(i)], n_epochs=2, policy_kwargs=policy_kwargs)

        # Initialize the self-play with two random models
        model.save("./model_directory/" + str(i) + "/Number_0_team_" + str(i))
        model.save("./model_directory/" + str(i) + "/Number_1_team_" + str(i))

        # Start the counter where it left off for training.
        try:
            counter[str(i)] = start_training()
            print("starting from" + str(start_training()))
        except:
            counter[str(i)] = 1
            print("starting from" + str(1))

    # If there is a pickle file that describes our enviroment dictionaries, lets load them and continue training.
    if os.path.isfile('thread_list_elo.pickle') and os.path.isfile('performance_dict.pickle'):

        with open('performance_dict.pickle', 'rb') as handle:
            performance_dict = pickle.load(handle)

        with open('thread_list_elo.pickle', 'rb') as handle:
            thread_list_elo = pickle.load(handle)

        with open('extra_data.pickle', 'rb') as handle:
            extra_data = pickle.load(handle)


    # Inititialize a number to start counting tournaments
    num = 1

    while num < number_tournaments:

        # This loop defines which team is training
        for team in range(how_many_teams_do_you_have):

            print("Starting New Round")

            move_on_check = True
            train_again_flag = True
            time_step_trained = 0
            running_count = 0

            while move_on_check:

                # We use this to track max iterations on one round.
                running_count += 1

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
                model.load("./model_directory/" + str(team) + "/Number_" + str(counter[str(team)]) + "_team_" + str(team), env=env_dict[str(team)])

                # Start the learning processes
                model.learn(total_timesteps=train_for_time_steps)

                model.save("./model_directory/" + str(team) + "/Number_" + str(counter[str(team)]) + "_team_" + str(team))

                thread_list_elo[str(team)]["Number_" + str(counter[str(team)] ) + "_team_" + str(team)] = 1200

                # Create a vectorized enviroment to do parallel processing
                if train_again_flag == True:

                    temp_elo_list = {"Number_" + str(counter[str(team)]) + "_team_" + str(team): thread_list_elo[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)]}

                    temp_elo_list["Number_" + str(counter[str(team)]-1) + "_team_" + str(team)] = thread_list_elo[str(team)]["Number_" + str(counter[str(team)]-1) + "_team_" + str(team)]

                    performance_dictionary, extra_info = eval_list_of_models(temp_elo_list, env_dict[str(team)], model,
                                                             num_evals, team)
                else:
                    performance_dictionary, extra_info = eval_list_of_models(thread_list_elo[str(team)], env_dict[str(team)], model,
                                                             num_evals, team)



                thread_list_elo_active = update_elo(performance_dictionary, thread_list_elo[str(team)], team)
                thread_list_elo[str(team)] = thread_list_elo_active

                extra_data[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)][0] += train_for_time_steps
                extra_data[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)][1] = extra_info[0]

                for keys in performance_dictionary:
                    performance_dict[str(team)][keys]["Round_" + str(counter[str(team)])] = performance_dictionary[keys]["rewards"]

                denominator = performance_dictionary["Number_" + str(counter[str(team)] - 1) + "_team_" + str(team)]["rewards"] + performance_dictionary["Number_" + str(counter[str(team)]) + "_team_" + str(team)]["rewards"]

                if performance_dictionary["Number_" + str(counter[str(team)]) + "_team_" + str(team)]["rewards"]/denominator> performance_dictionary["Number_" + str(counter[str(team)]-1) + "_team_" + str(team)]["rewards"]/denominator * move_on_threshold or running_count>100:
                    counter[str(team)] += 1

                    model.save( "./model_directory/" + str(team) + "/Number_" + str(counter[str(team)]) + "_team_" + str(team))

                    #Create a dummy one first then move
                    performance_dict[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)] = {"Round_" + str(1) : 0}
                    for round_pre_fill in range(2,counter[str(team)]):
                        performance_dict[str(team)]["Number_" + str(counter[str(team)])+"_team_" + str(team)]["Round_" + str(round_pre_fill)] = 0

                    thread_list_elo[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)] = thread_list_elo[str(team)]["Number_" + str(counter[str(team)]-1) + "_team_" + str(team)]
                    extra_data[str(team)]["Number_" + str(counter[str(team)]) + "_team_" + str(team)] = [0,0]

                    if terminal_state > 2:
                        terminal_state = terminal_state - 1
                    move_on_check = False
                    train_again_flag = True
                else:
                    train_again_flag = True

                save_leader_board(thread_list_elo)
                save_reward_board([performance_dict])
                save_extra_board(extra_data)

                with open('thread_list_elo.pickle', 'wb') as handle:
                    pickle.dump(thread_list_elo, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('performance_dict.pickle', 'wb') as handle:
                    pickle.dump(performance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open('extra_data.pickle', 'wb') as handle:
                    pickle.dump(extra_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        num += 1







