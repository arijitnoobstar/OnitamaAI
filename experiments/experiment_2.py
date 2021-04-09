# access Train.py in parent folder and set relative folder to parent folder for data saving

import os
import sys
os.chdir("..")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from Train import *

"""
The purpose of this experiment is to observe the difference
in training with different learning rates of 0.1, 0.01, 0.001
"""

# EXPERIMENT 2 PARAMETERS, NOTE: val_branch_actions is used

ep_num = 5000
plot_every = 1000
moving_average = 50
minimax_boost = 1

onitama_deeprl_train("train", "DDPG", ep_num, "lr_0.1", "minimax", 1, discount_rate = 0.99, 
            lr_actor = 0.1, lr_critic = 0.1, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
            num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
            epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
            use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
            reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
            architecture = "val_branch_actions", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "lr_0.01", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.01, lr_critic = 0.01, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "val_branch_actions", moving_average = moving_average, verbose = False)

onitama_deeprl_train("train", "DDPG", ep_num, "lr_0.001", "minimax", 1, discount_rate = 0.99, 
          lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
          num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
          epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
          use_competing_AI_replay = False, win_loss_mem_size = 1000, desired_win_ratio = 0.6, use_hardcoded_cards = True,
          reward_mode = "simple_reward", minimax_boost = minimax_boost, mcts_boost = 5000, plot_every = plot_every, win_loss_queue_size = 100,
          architecture = "val_branch_actions", moving_average = moving_average, verbose = False)