from std_imports import *
from Onitama_source import *
from MCTS import *
from std_imports import *
from replay_buffer import *
from NN import *
from Agent import *

# This code implements the training process of the Onitama deep RL agent
# The agent plays against a competing AI (either minimax or MCTS) to automate the learning process
# This use of a bot/AI as a competitor allows for no human labour or intervention

def onitama_deeprl_train(mode, model, episodes, training_name, competing_AI_mode, competing_AI_strength, discount_rate = 0.99, 
              lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 48, 
              num_actions = 40, max_mem_size = 1000000, batch_size = 512, epsilon = 1,
              epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = False,
              use_competing_AI_replay = True, win_loss_mem_size = 1000, desired_win_ratio = 0.9, use_hardcoded_cards = True):
  
  # create the environment
  game = Onitama(verbose = False)

  # calculate the epsilon decay based on number of episodes for training (for a linearly decreasing epsilon)
  epsilon_decay = (epsilon - epsilon_min) / episodes

  # adjust card_input shape if hardcoded version is used
  if use_hardcoded_cards:
    card_input_shape = 10

  # create an instance of the agent
  agent = Agent(model, discount_rate, lr_actor, lr_critic, tau, board_input_shape, card_input_shape, num_actions, 
                 max_mem_size, batch_size, epsilon, epsilon_decay, epsilon_min, update_target, val_constant, 
                 training_name)
  
  # load model if testing mode
  if mode == "test":

    agent.load_all_models()

  else:
    try:
      os.mkdir("Training_Plots/" + training_name  + "/")
    except:
        shutil.rmtree("Training_Plots/" + training_name  + "/")
        os.mkdir("Training_Plots/" + training_name  + "/")

  # NOTE that one turn here refers to one turn of the Deep RL agent (not inclusive of the turn by competing AI (either minimax or mcts))
  total_turns = 0
  
  # to add to depth of minimax
  AI_strength_boost = 0

  # binary list to store win loss
  win_loss_list = [0 for x in range(win_loss_mem_size)]

  # win loss counter
  win_loss_counter = 0

  # win list to store episode numbers of when the agent wins against the AI
  win_list =[]

  # lists to store losses 
  if model == "DDPG":
    
    actor_training_loss_list = []
    actor_val_loss_list = []
    actor_loss_list = []
    critic_loss_list = []
  
  elif model == "D3QN":

    training_loss_list = []
    val_loss_list = []
    total_loss_list = []

  for episode in range(episodes):

    print("Episode: {}, Total Turns: {}".format(episode + 1, total_turns))

    if use_hardcoded_cards:
      game.start_hardcoded_game()
    else:
      game.start_game()

    # let the red player be the competing AI while the blue player be the Deep RL agent
    if game.whose_turn == 'red':
      # if red gets to go first, then let the competing AI make a move first
      if competing_AI_mode.lower() == 'minimax':
        # print("{} : opponent".format(game.whose_turn))
        # first turn for minimax AI
        game.turn_minimax(minimax_depth = competing_AI_strength + AI_strength_boost)
      else:
        # first turn for MCTS AI (create the AI object first) 
        mcts_ai = mcts(iterationLimit = competing_AI_strength + AI_strength_boost, verbose = 0)
        game.turn_mcts(mcts_object = mcts_ai)
    else:
      # if blue gets to go first, create the MCTS object first (if mcts is used as the competing AI)
      if competing_AI_mode.lower() == "mcts":
        mcts_ai = mcts(iterationLimit = competing_AI_strength + AI_strength_boost, verbose = 0)

    # regardless of whether red went first, now we record the board_state and card_state in both perspectives
    agent_board_state, agent_card_state = game.return_NN_state(perspective = 'blue', use_hardcoded = use_hardcoded_cards)
    if use_competing_AI_replay:
      opponent_board_state_prime, opponent_card_state_prime = game.return_NN_state(perspective = 'red', use_hardcoded = use_hardcoded_cards)

    # boolean to indicate if episode has terminated
    is_done = 0

    # run episode until it terminates
    while is_done == 0:

      # boolean to check if action spit out is valid
      is_valid_action = 0
      # counter to keep track of consecutive turns where Deep RL AI chose an invalid move
      is_invalid_counter = 0
      # get a list of valid actions (1 for valid and 0 otherwise)
      val_actions_target = game.check_NN_valid_move_space()

      while is_valid_action == 0:

        # if hand_of_god is true, then force the AI to only select valid actions
        # get the actions (integer from 0 - 39) to be taken based on the board and card states
        if hand_of_god: 
          action_probs, action = agent.select_action("train", agent_board_state, agent_card_state, val_actions_target)
      
        else:
          action_probs, action = agent.select_action("train", agent_board_state, agent_card_state)
          
        # play the action if it is valid, otherwise the board and card states remain the same
        if game.check_NN_valid_move(move_index = action):
          # print("{} : deep rl".format(game.whose_turn))
          game.turn_deeprl(move_index = action)
          is_valid_action = 1
        # else:
        #   # if it is invalid --> apply hand of god to get the board_state_prime and card_state_prime
        #   action = agent.select_action("train", board_state, card_state, val_actions_target)
        #   game.turn_deeprl(move_index = action)

        # record the state prime observations
        agent_board_state_prime, agent_card_state_prime = game.return_NN_state(perspective = 'blue', use_hardcoded = use_hardcoded_cards)
        if use_competing_AI_replay:
          opponent_board_state, opponent_card_state = game.return_NN_state(perspective = 'red', use_hardcoded = use_hardcoded_cards)

        if game.blue_win:
          is_done = 1
          win_loss_counter += 1
          index = win_loss_counter % win_loss_mem_size
          win_loss_list[index] = 1
          win_list.append(episode)
          print("deep rl wins##################################################################")

        # get the reward (multiplied by -1 as the original negative rewards favour the blue (deep RL agent) player)
        reward = -game.eval_board_state()

        # add a further penalty if an invalid piece was chosen
        if not is_valid_action:
          reward -= invalid_penalty
          is_invalid_counter += 1
          print("Turn {}: Agent Chose wrongly for {} consecutive tries".format(total_turns, is_invalid_counter))

        # store experience in replay memory
        agent.store_memory(agent_board_state, agent_card_state, action_probs, val_actions_target, reward,
         agent_board_state_prime, agent_card_state_prime, is_done)

        # apply gradients for learning to reduce loss and append losses to lists
        try:
          if model == "DDPG" and mode == "train":
            actor_training_loss, actor_val_loss, actor_loss, critic_loss = agent.apply_gradients_DDPG()
            actor_training_loss_list.append(actor_training_loss)
            actor_val_loss_list.append(actor_val_loss)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
          elif model == "D3QN" and mode == "train":
            training_loss, val_loss, total_loss = agent.apply_gradients_D3QN()
            training_loss_list.append(training_loss)
            val_loss_list.append(val_loss)
            total_loss_list.append(total_loss)
        except:
          pass
        total_turns += 1

      # Let the competing AI make its responsive turn if game has not terminated
      if not game.isTerminal():
        if competing_AI_mode.lower() == 'minimax':
          # next turn for minimax AI
          # print("{} : opponent".format(game.whose_turn))
          game.turn_minimax(minimax_depth = competing_AI_strength, return_move_index = True)
        else:
          # next turn for MCTS AI
          game.turn_mcts(mcts_object = mcts_ai)

      # check if red wins (opponent AI)
      if game.red_win:
        is_done = 1
        win_loss_counter += 1
        index = win_loss_counter % win_loss_mem_size
        win_loss_list[index] = 0
        print("deep rl loses")
        # print(-1 * game.eval_board_state())

      # record valid actions for opponent
      val_actions_target = game.check_NN_valid_move_space()

      # re-record the board_state and card_state
      agent_board_state, agent_card_state = game.return_NN_state(perspective = 'blue', use_hardcoded = use_hardcoded_cards)
      if use_competing_AI_replay:
        opponent_board_state_prime, opponent_card_state_prime = game.return_NN_state(perspective = 'red', use_hardcoded = use_hardcoded_cards)

      # add in experience replay of competing AI if specified
      if use_competing_AI_replay:

        reward = game.eval_board_state() # no -ve needed as the perspective is from the competing AI

        # extract the action taken by competing AI in terms of move_index (0-39)
        if competing_AI_mode.lower() == 'minimax':
          action = game.selected_move_index
        else:
          action = mcts_ai.move_index

        # obtain one hot encoded action
        action_one_hot = np.zeros(40)
        action_one_hot[action] = 1

        # store experience in replay (note that the opponent's perspective is used to preserve the order of the states)
        agent.store_memory(opponent_board_state, opponent_card_state, action_one_hot, val_actions_target,
        reward, opponent_board_state_prime, opponent_card_state_prime, is_done)

      # increment total_turns
      total_turns += 1
    
    # plot losses
    if model == "DDPG" and mode == "train" and episode % 50 == 0 and episode != 0: 

      plt.title("DDPG Actor Training Loss")
      plot_1 = sns.lineplot(data = np.array(actor_training_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_ddpg_actor_training_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

      plt.title("DDPG Actor Validity Loss")
      plot_2 = sns.lineplot(data = np.array(actor_val_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_ddpg_actor_val_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

      plt.title("DDPG Actor Loss")
      plot_3 = sns.lineplot(data = np.array(actor_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_ddpg_actor_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

      plt.title("DDPG Critic Loss")
      plot_4 = sns.lineplot(data = np.array(critic_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_ddpg_critic_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

    elif model == "D3QN" and episode % 50 == 0 and episode != 0:

      plt.title("D3QN Training Loss")
      plot_1 = sns.lineplot(data = np.array(training_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_d3qn_training_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

      plt.title("D3QN Validity Loss")
      plot_2 = sns.lineplot(data = np.array(val_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_d3qn_validity_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

      plt.title("D3QN Loss")
      plot_3 = sns.lineplot(data = np.array(total_loss_list))
      plt.ylabel("Loss")
      plt.xlabel("Number of episodes")
      plt.savefig("Training_Plots/" + training_name  + "/" + training_name + f'_d3qn_loss_episode_{episode}.pdf', 
            bbox_inches = 'tight')
      plt.close()
      plt.show() 

    # improve opponent if model has desired performance against current opponent
    if sum(win_loss_list) / win_loss_mem_size >= desired_win_ratio and mode == 'train':

      # reset win lost list and counter
      win_loss_list = [0 for x in range(win_loss_mem_size)]
      win_loss_counter = 0

      # save model
      agent.save_all_models()

      if competing_AI_mode.lower() == 'minimax':

        AI_strength_boost += 1
      
      elif competing_AI_mode.lower() == "mcts":

        AI_strength_boost += 5000

    # reset the game
    game.reset()

  # print win list
  print(win_list)
  print("Agent won {} games out of {}".format(len(win_list), episodes))

if __name__ == "__main__":
  onitama_deeprl_train("train", "DDPG", 5000, "testing_integration", "minimax", 1, discount_rate = 0.99, 
              lr_actor = 0.001, lr_critic = 0.001, tau = 0.005, board_input_shape = [4, 5, 5], card_input_shape = 10, 
              num_actions = 40, max_mem_size = 1000000, batch_size = 128, epsilon = 1,
              epsilon_min = 0.01, update_target = None, val_constant = 10, invalid_penalty = 500, hand_of_god = True,
              use_competing_AI_replay = True, win_loss_mem_size = 1000, desired_win_ratio = 0.9, use_hardcoded_cards = True)