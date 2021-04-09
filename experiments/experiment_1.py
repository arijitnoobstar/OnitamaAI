# access Train.py in parent folder and set relative folder to parent folder for data saving
import os
import sys
os.chdir("..")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from Train import *

"""
The purpose of this experiment is to run and validate the performance between 
6 different architectures to tackle validity of moves. The aim is to find a good balance of
valid moves and performance
"""

# EXPERIMENT 1 PARAMETERS, NOTE: Boost is set to 0

ep_num = 20000
plot_every = 4000
moving_average = 50
minimax_boost = 0

onitama_deeprl_train("train", "DDPG", ep_num, "val_after_actions", "minimax", 1, discount_rate = 0.99, 
            lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
            num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
            epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
            use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
            reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
            architecture = "val_after_actions", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "val_after_actions_multiply", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "val_after_actions_multiply", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "val_branch_actions", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "val_branch_actions", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "val_branch_actions_multiply", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "val_branch_actions_multiply", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "actions_after_val", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "actions_after_val", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "actions_after_val_multiply", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "actions_after_val_multiply", moving_average = moving_average, verbose = False)  
