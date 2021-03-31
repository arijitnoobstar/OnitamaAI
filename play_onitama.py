from std_imports import *
from Onitama_source import *
from MCTS import *

def play_onitama(game_mode, first_move = None, verbose = 1, minimax_depth = None, minimax_depth_red = None, minimax_depth_blue = None, aivai_turns = 500, timeLimit=None, iterationLimit=None, iteration_red = None, iteration_blue = None, mcts_efficiency = "space", parallel = None):

  # create game instance
  game = Onitama(verbose = verbose)

  # initialise game
  game.start_game(first_move = first_move)

  # run different game modes: pvp, ai v ai(minimax), ai v p
  if game_mode == 'p v p':
    # play game till one player wins
    while game.blue_win != True and game.red_win != True:
      game.turn_pvp()

  elif game_mode == 'minimax v minimax':
    # pitch minimax algo against each other
    if minimax_depth_red == None and minimax_depth != None:
      minimax_depth_red = minimax_depth
    if minimax_depth_blue == None and minimax_depth != None:
      minimax_depth_blue = minimax_depth
    while game.blue_win != True and game.red_win != True and game.number_of_turns < aivai_turns:
      start_time = time.time()
      if game.whose_turn == "blue":
        game.turn_minimax(minimax_depth = minimax_depth_blue, parallel = parallel)
      else:
        game.turn_minimax(minimax_depth = minimax_depth_red, parallel = parallel)
      end_time = time.time() - start_time
      print("Turn {} Completed in {:.1f}s".format(game.number_of_turns - 1, end_time))

  elif game_mode == "minimax v p":
    # Minimax vs Player, let AI be red player
    while game.blue_win != True and game.red_win != True:
      if game.whose_turn == "blue":
        game.turn_pvp()
      else:
        game.turn_minimax(minimax_depth = minimax_depth, parallel = parallel)


  elif game_mode.lower() == "mcts v p":
    # MCTS vs Player, let AI be red player
    mcts_ai = mcts(iterationLimit = iterationLimit, timeLimit = timeLimit, verbose = verbose, efficiency = mcts_efficiency)
    while game.blue_win != True and game.red_win != True:
      if game.whose_turn == "blue":
        game.turn_pvp()
      else:
        game.turn_mcts(mcts_object = mcts_ai)
        
  elif game_mode.lower() == "mcts v minimax":
    # MCTS vs Minimax, let Minimax be red player
    mcts_ai = mcts(iterationLimit = iterationLimit, timeLimit = timeLimit, verbose = verbose, efficiency = mcts_efficiency)
    while game.blue_win != True and game.red_win != True:
      start_time = time.time()
      if game.whose_turn == "blue":
        game.turn_mcts(mcts_object = mcts_ai)
      else:
        game.turn_minimax(minimax_depth = minimax_depth, parallel = parallel)
      end_time = time.time() - start_time
      print("Turn {} Completed in {:.1f}s".format(game.number_of_turns - 1, end_time))

  elif game_mode.lower() == "mcts v mcts":
    # MCTS vs MCTS
    mcts_blue = mcts(iterationLimit = iteration_blue, verbose = verbose, efficiency = mcts_efficiency)
    mcts_red = mcts(iterationLimit = iteration_red, verbose = verbose, efficiency = mcts_efficiency)
    while game.blue_win != True and game.red_win != True:
      start_time = time.time()
      if game.whose_turn == "blue":
        game.turn_mcts(mcts_object = mcts_blue)
      else:
        game.turn_mcts(mcts_object = mcts_red)
      end_time = time.time() - start_time
      print("Turn {} Completed in {:.1f}s".format(game.number_of_turns - 1, end_time))

  if verbose:
    game.show_game_state

  if game.blue_win == True:
    print("Blue won!")
    return game
  elif game.red_win == True:
    print("Red won!")
    return game
  else:
    print("Its a draw! Yall good at the game...")
    return game



def run_onitama_experiment(games, equal_first_moves = True, path = None, game_mode = "minimax v minimax", minimax_depth = None , minimax_depth_red = None, minimax_depth_blue = None, timeLimit=None, iterationLimit = None, verbose = 0, iteration_red = None, iteration_blue = None, mcts_efficiency = "space", parallel = None):
  winner_list = []
  number_of_turns_list = []
  method_of_win_list = []
  game_list = []
  games = games

  if games%2 != 0 and equal_first_moves:
    raise Exception("Number of games must be even for equal number of first moves for each player")


  for i in range(games):
    print("Game {}:".format(i+1), end = ' ')
    # for equal first moves, both players get the same number of chances to start first
    if equal_first_moves:
      if i%2 == 0:
        game = play_onitama(game_mode = game_mode, first_move = "blue", minimax_depth = minimax_depth , mcts_efficiency = "space", parallel = None, minimax_depth_red = minimax_depth_red, minimax_depth_blue = minimax_depth_blue, timeLimit=timeLimit, iterationLimit = iterationLimit, iteration_red = iteration_red, iteration_blue = iteration_blue, verbose = verbose)
      else:
        game = play_onitama(game_mode = game_mode, first_move = "red", minimax_depth = minimax_depth , mcts_efficiency = "space", parallel = None, minimax_depth_red = minimax_depth_red, minimax_depth_blue = minimax_depth_blue, timeLimit=timeLimit, iterationLimit = iterationLimit, iteration_red = iteration_red, iteration_blue = iteration_blue, verbose = verbose)
    # only red starts first
    else:
      game = play_onitama(game_mode = game_mode, first_move = "blue", minimax_depth = minimax_depth , mcts_efficiency = "space", parallel = None, minimax_depth_red = minimax_depth_red, minimax_depth_blue = minimax_depth_blue, timeLimit=timeLimit, iterationLimit = iterationLimit, iteration_red = iteration_red, iteration_blue = iteration_blue, verbose = verbose)
    
    game_list.append(game)
    if game.check_win() == False:
      continue
    number_of_turns_list.append(game.number_of_turns)
    if game.blue_win == True:
      winner_list.append("blue")
      if game.piece_state['R'] == -1:
        method_of_win_list.append("destroy")
      else:
        method_of_win_list.append("conquer")
    else:
      winner_list.append("red")
      if game.piece_state['B'] == -1:
        method_of_win_list.append("destroy")
      else:
        method_of_win_list.append("conquer")

  red_win_count = winner_list.count('red')
  blue_win_count = len(winner_list) - red_win_count

  destroy_count = method_of_win_list.count('destroy')
  conquer_count = len(method_of_win_list) - destroy_count

  red_destroy_count = 0
  red_conquer_count = 0
  blue_destroy_count = 0
  blue_conquer_count = 0

  red_number_of_turns_list = []
  blue_number_of_turns_list = []

  for count, x in enumerate(winner_list):
    if x == 'red':
      red_number_of_turns_list.append(number_of_turns_list[count])
      if method_of_win_list[count] == 'destroy':
        red_destroy_count += 1
      else:
        red_conquer_count += 1
    else:
      blue_number_of_turns_list.append(number_of_turns_list[count])
      if method_of_win_list[count] == 'destroy':
        blue_destroy_count += 1
      else:
        blue_conquer_count += 1

  red_player =  game_mode.split()[2]
  blue_player =  game_mode.split()[0]

  if red_player == "minimax":
    if minimax_depth_red != None:
      red_player += " (depth {})".format(minimax_depth_red)
    else:
      red_player += " (depth {})".format(minimax_depth)
  elif red_player == "mcts":
    if iteration_red != None:
      red_player += " ({} iterations)".format(iteration_red)
    else:
      red_player += " ({} iterations)".format(iterationLimit)
  
  if blue_player == "minimax":
    if minimax_depth_blue != None:
      blue_player += " (depth {})".format(minimax_depth_blue)
    else:
      blue_player += " (depth {})".format(minimax_depth)
  elif blue_player == "mcts":
    if iteration_blue != None:
      blue_player += " ({} iterations)".format(iteration_blue)
    else:
      blue_player += " ({} iterations)".format(iterationLimit)

  plt.plot(np.arange(1,len(red_number_of_turns_list) + 1), red_number_of_turns_list, label = "Red Wins", color = 'red')
  plt.plot(np.arange(1,len(blue_number_of_turns_list) + 1), blue_number_of_turns_list, label = "Blue Wins", color = 'blue')
  plt.xlabel("Game Number")
  plt.ylabel("Number of Turns Played")
  plt.title("Number of Turns Played per Game")
  plt.legend()
  if path != None:
    plt.savefig(path + "/plot_1.png")
  else:
    plt.show()  
  plt.close()

  plt.bar([0,1], [red_win_count, blue_win_count], color = ['red','blue'])
  plt.xticks([0,1], (red_player, blue_player))
  plt.xlabel("Player")
  plt.ylabel("Number of Wins")
  plt.title("Number of Wins per Player")
  if path != None:
    plt.savefig(path + "/plot_2.png")
  else:
    plt.show()  
  plt.close()

  ind = np.arange(2)    # the x locations for the groups
  width = 0.35       # the width of the bars: can also be len(x) sequence

  p1 = plt.bar(ind, [blue_destroy_count,blue_conquer_count], width, color = 'blue')
  p2 = plt.bar(ind, [red_destroy_count,red_conquer_count], width,
               bottom=[blue_destroy_count,blue_conquer_count], color = 'red')

  plt.ylabel('Frequency')
  plt.title('Frequecy of Method games are won')
  plt.xticks(ind, ('Destroy', 'Conquer'))
  plt.legend((p1[0], p2[0]), (blue_player, red_player))
  if path != None:
    plt.savefig(path + "/plot_3.png")
  else:
    plt.show()
  plt.close()
  try:
    print("Average Number of Turns (Total): ", sum(number_of_turns_list)/len(number_of_turns_list))
  except:
    pass
  try:
    print("Average Number of Turns (Red Wins): ", sum(red_number_of_turns_list)/len(red_number_of_turns_list))
  except:
    pass
  try:
    print("Average Number of Turns (Blue Wins): ", sum(blue_number_of_turns_list)/len(blue_number_of_turns_list))
  except:
    pass

  return game_list

game_list_list = []