# Standard Imports
from std_imports import *
from MCTS import *
# The onitama class represents the board game state at any instant of gameplay

class Onitama:

  """ Represents the board game state of the game Onitama """

  def __init__(self, verbose = 1):

    """ Class constructor, all game state variables are defined here"""

    # Set to 1 to print out gameplay, 0 otherwise
    self.verbose = verbose

    # The board is a 2D list of the 5x5 grid representation (row --> i & column --> j)
    # b1,b2,b3,b4 and r1,r2,r3,r4 refer to the blue and red team pawns
    # B and R refer to the blue and red team masters

    self.board = [['b1','b2','B','b3','b4'], # row 0
                  ['•','•','•','•','•'], # row 1
                  ['•','•','•','•','•'], # row 2
                  ['•','•','•','•','•'], #row 3
                  ['r1','r2','R','r3','r4']] # row 4

    # save the position of each piece in a dictionary. This is used to translate piece names to their position and vice-versa
    # in the event the player is out of the game, the value should be saved as -1. e.g. if b3 is out of the game, it should show 'b3': -1

    self.piece_state = {'b1':[0,0], 'b2': [0,1],'b3':[0,3], 'b4': [0,4], 'B': [0,2], 'r1':[4,0], 'r2': [4,1],'r3':[4,3], 'r4': [4,4], 'R': [4,2]}
    self.blue_pieces = ['b1', 'b2', 'b3', 'b4', 'B'] # list of blue pieces 
    self.red_pieces = ['r1', 'r2', 'r3', 'r4', 'R'] # list of red pieces

    # The cards dictionary store the moves for each one of the 16 cards in onitama. The key of each card is another dictionary with the move name and move coordinate change
    # The movement is stored in (i,-j) format for a BLUE player and (-i,j) for a RED player wrt the board.
    # NOTE that for a red player, i must be multiplied by -1 when implementing into the board; while for a blue player, j must be multiplied by -1
    # hence in the cards dictionary a positive first index means moving forward and a positive second index means moving to the right for ANY player
    
    # The moves are given names (A, B, C & D) which we shall refer to as the variable 'choice' in the move method of this class
    self.cards = {
        'tiger' : {'A' : (2,0), 'B' : (-1,0)},
        'dragon' : {'A' : (1,2), 'B' : (1,-2), 'C' : (-1,1), 'D' : (-1,-1)},
        'frog' : {'A' : (0,-2), 'B' : (1,-1), 'C' : (-1,1)},
        'rabbit' : {'A' : (1,1), 'B' : (0,2), 'C' : (-1,-1)},
        'crab' : {'A' : (1,0), 'B' : (0,2), 'C' : (0,-2)},
        'elephant' : {'A' : (1,1), 'B' : (1,-1), 'C' : (0,1), 'D' : (0,-1)},
        'goose' : {'A' : (1,-1), 'B' : (0,-1), 'C' : (0,1), 'D' : (-1,1)},
        'rooster' : {'A' : (-1,-1), 'B' : (0,-1), 'C' : (0,1), 'D' : (1,1)},
        'monkey' : {'A' : (-1,-1), 'B' : (-1,1), 'C' : (1,-1), 'D' : (1,1)},
        'mantis' : {'A' : (-1,0), 'B' : (1,1), 'C' : (1,-1)},
        'horse' : {'A' : (1,0), 'B' : (-1,0), 'C' : (0,-1)},
        'ox' : {'A' : (1,0), 'B' : (0,1), 'C' : (-1,0)},
        'crane' : {'A' : (-1,1), 'B' : (-1,-1), 'C' : (1,0)},
        'boar' : {'A' : (1,0), 'B' : (0,1), 'C' : (0,-1)},
        'eel' : {'A' : (1,-1), 'B' : (-1,-1), 'C' : (0,1)},
        'cobra' : {'A' : (0,-1), 'B' : (1,1), 'C' : (-1,1)}
    }


    # as the decision on who gets to go first is dependent on the colour of the card on the side, this dictionary saves that information

    self.card_colour = {
        'tiger' : 'blue',
        'dragon' : 'red',
        'frog' : 'red',
        'rabbit' : 'blue',
        'crab' : 'blue',
        'elephant' : 'red',
        'goose' : 'blue',
        'rooster' : 'red',
        'monkey' : 'blue',
        'mantis' : 'red',
        'horse' : 'red',
        'ox' : 'blue',
        'crane' : 'blue',
        'boar' : 'red',
        'eel' : 'blue',
        'cobra' : 'red'
    }

    # Create empty representations of the position of the cards available to each player and the card on the side

    self.blue_cards = [] 
    self.red_cards = []
    self.side_card = []

    # Initialise a few more attributes that can change depending on the gameplay
    
    self.whose_turn = None # either 'blue' or 'red' during gameplay
    self.number_of_turns = 0 # increment by 1 after each player makes a move
    self.blue_win = False
    self.red_win = False
    self.selected_pawn = None # stores selected pawn for a move in a turn 
    self.selected_card = None # stores selected card for a move in a turn
    self.selected_move = None # stores selected move chosen for a turn
    self.undo_turn = None # turn to undo to for pvp

    # Initialise copies of the above for history keeping purposes

    self.whose_turn_log = []
    self.number_of_turns_log = []
    self.blue_win_log = []
    self.red_win_log = []
    self.blue_cards_log = []
    self.red_cards_log = []
    self.side_card_log = []
    self.board_log = []
    self.piece_state_log = []
    self.selected_pawn_log = []
    self.selected_card_log = []
    self.selected_move_log = []
    
    # Initialise features need for minimax algo
    self.depth = None # depth of tree for minimax
    self.best_selected_pawn = None # stores best move from minimax
    self.best_selected_card = None # stores best move from minimax
    self.best_selected_move = None # stores best move from minimax

  def reset(self):

    """ resets the game state so a new game can be played in the same object """

    self.board = [['b1','b2','B','b3','b4'], 
                  ['•','•','•','•','•'], 
                  ['•','•','•','•','•'],
                  ['•','•','•','•','•'], 
                  ['r1','r2','R','r3','r4']]
    self.piece_state = {'b1':[0,0], 'b2': [0,1],'b3':[0,3], 'b4': [0,4], 'B': [0,2], 'r1':[4,0], 'r2': [4,1],'r3':[4,3], 'r4': [4,4], 'R': [4,2]}
    self.blue_pieces = ['b1', 'b2', 'b3', 'b4', 'B'] 
    self.red_pieces = ['r1', 'r2', 'r3', 'r4', 'R'] 
    self.blue_cards = [] 
    self.red_cards = []
    self.side_card = []
    self.whose_turn = None
    self.number_of_turns = 0 
    self.blue_win = False
    self.red_win = False
    self.selected_pawn = None 
    self.selected_card = None 
    self.selected_move = None 
    self.undo_turn = None 
    self.whose_turn_log = []
    self.number_of_turns_log = []
    self.blue_win_log = []
    self.red_win_log = []
    self.blue_cards_log = []
    self.red_cards_log = []
    self.side_card_log = []
    self.board_log = []
    self.piece_state_log = []
    self.selected_pawn_log = []
    self.selected_card_log = []
    self.selected_move_log = []
    self.depth = None 
    self.best_selected_pawn = None 
    self.best_selected_card = None 
    self.best_selected_move = None 

  def start_hardcoded_game(self, first_move = None):

    """ Starts the game by 'shuffling' 5 HARDCODED cards into the game, which determines the starting player """

    self.hardcoded_cards = {
        'dragon' : {'A' : (1,2), 'B' : (1,-2), 'C' : (-1,1), 'D' : (-1,-1)},
        'elephant' : {'A' : (1,1), 'B' : (1,-1), 'C' : (0,1), 'D' : (0,-1)},
        'goose' : {'A' : (1,-1), 'B' : (0,-1), 'C' : (0,1), 'D' : (-1,1)},
        'rooster' : {'A' : (-1,-1), 'B' : (0,-1), 'C' : (0,1), 'D' : (1,1)},
        'monkey' : {'A' : (-1,-1), 'B' : (-1,1), 'C' : (1,-1), 'D' : (1,1)},
    }

    self.five_cards = []

    # for loop randomly selects 5 distinct cards from hardcoded_cards and puts them into self.five_cards list
    temp_cards = copy.deepcopy(self.hardcoded_cards)
    for i in range(5):
      # if a first move is specified, keep selecting a side_Card until it matches the turn of the player specified for first move
      if i == 0 and first_move != None:
        temp = None
        while temp != first_move.lower():
          potential_side_card = random.choice(list(temp_cards.keys()))
          temp = self.card_colour[potential_side_card]
        self.five_cards.append(potential_side_card)
      else:
        self.five_cards.append(random.choice(list(temp_cards.keys()))) # randomly select a card into our five cards
      temp_cards.pop(self.five_cards[i]) # remove that card from temp_cards so no repetitions can occur

    # assign the five cards to both players and the side
    self.blue_cards.extend(self.five_cards[1:3])
    self.red_cards.extend(self.five_cards[3:5])
    self.side_card.append(self.five_cards[0])

    # determine the starting player
    if self.card_colour[self.side_card[0]] == 'blue':
      self.whose_turn = 'blue'
    else:
      self.whose_turn = 'red'

    # Save the history for TURN 0 (settings for initial starting position)
    self.save_history()
    # increment the number of turns to TURN 1
    self.number_of_turns += 1

  def start_game(self, first_move = None):

    """ Starts the game by 'shuffling' 5 cards into the game, which determines the starting player """

    self.five_cards = []

    # for loop randomly selects 5 distinct cards from self.cards and puts them into self.five_cards list
    temp_cards = copy.deepcopy(self.cards)
    for i in range(5):
      # if a first move is specified, keep selecting a side_Card until it matches the turn of the player specified for first move
      if i == 0 and first_move != None:
        temp = None
        while temp != first_move.lower():
          potential_side_card = random.choice(list(temp_cards.keys()))
          temp = self.card_colour[potential_side_card]
        self.five_cards.append(potential_side_card)
      else:
        self.five_cards.append(random.choice(list(temp_cards.keys()))) # randomly select a card into our five cards
      temp_cards.pop(self.five_cards[i]) # remove that card from temp_cards so no repetitions can occur

    # assign the five cards to both players and the side
    self.blue_cards.extend(self.five_cards[1:3])
    self.red_cards.extend(self.five_cards[3:5])
    self.side_card.append(self.five_cards[0])

    # determine the starting player
    if self.card_colour[self.side_card[0]] == 'blue':
      self.whose_turn = 'blue'
    else:
      self.whose_turn = 'red'

    # Save the history for TURN 0 (settings for initial starting position)
    self.save_history()
    # increment the number of turns to TURN 1
    self.number_of_turns += 1
  
  @property
  def show_board(self):

    """ Shows board state neatly """

    print("current board:")
    print()
    # Loop between rows 
    for row in self.board:
      for elem in row:
        if len(elem) == 1:
          print(elem + " ", end = " ") # Print board state neatly for single lettered masters
        else:
          print(elem, end = " ") # Print board state neatly for pawns
      print()
    print()

  @property
  def show_player_moves(self):
    
    """ Shows the available moves to for player  """
    
    # If turn is blue
    if self.whose_turn == "blue": 
      print("player's (" + self.whose_turn + ") moves available:") # Print turn info
      print()

      left_card = self.blue_cards[0]
      right_card = self.blue_cards[1]
      print(left_card + ":" + " "*(16 - len(left_card)) + right_card + ":") # Print card name 

      moveset_left = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_left[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[left_card]: # Loop between choices 
        i_change, j_change = self.cards[left_card][choice]
        j_change *= -1  # Account for player's reference frame 
        moveset_left[2+i_change][2+j_change] = choice # Update array with move location

      moveset_right = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_right[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[right_card]: # Loop between choices 
        i_change, j_change = self.cards[right_card][choice]
        j_change *= -1  # Account for player's reference frame 
        moveset_right[2+i_change][2+j_change] = choice # Update array with move location
      for i in range(5):
        for j in range(5):
          print(moveset_left[i][j], end = " ")
        print("\t", end = ' ')
        for j in range(5):
          print(moveset_right[i][j], end = " ")
        print("\t")
      print("\n")

    else:
      print("player's (" + self.whose_turn + ") moves available:") # Print turn info
      print() 

      left_card = self.red_cards[0]
      right_card = self.red_cards[1]
      print(left_card + ":" + " "*(16 - len(left_card)) + right_card + ":") # Print card name 

      moveset_left = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_left[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[left_card]: # Loop between choices 
        i_change, j_change = self.cards[left_card][choice]
        i_change *= -1  # Account for player's reference frame 
        moveset_left[2+i_change][2+j_change] = choice # Update array with move location

      moveset_right = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_right[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[right_card]: # Loop between choices 
        i_change, j_change = self.cards[right_card][choice]
        i_change *= -1  # Account for player's reference frame 
        moveset_right[2+i_change][2+j_change] = choice # Update array with move location
      for i in range(5):
        for j in range(5):
          print(moveset_left[i][j], end = " ")
        print("\t", end = " ")
        for j in range(5):
          print(moveset_right[i][j], end = " ")
        print("\t")
      print("\n")
  
  @property
  def show_opponent_moves(self):
    
    """ Shows the available moves to for opponent  """
    
    # If turn is blue
    if self.whose_turn == "blue": 
      print("opponent (red) moves available:") # Print opponent's info
      print()

      left_card = self.red_cards[0]
      right_card = self.red_cards[1]
      print(left_card + ":" + " "*(16 - len(left_card)) + right_card + ":") # Print card name 

      moveset_left = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_left[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[left_card]: # Loop between choices 
        i_change, j_change = self.cards[left_card][choice]
        i_change *= -1  # Account for player's reference frame 
        moveset_left[2+i_change][2+j_change] = choice # Update array with move location

      moveset_right = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_right[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[right_card]: # Loop between choices 
        i_change, j_change = self.cards[right_card][choice]
        i_change *= -1  # Account for player's reference frame 
        moveset_right[2+i_change][2+j_change] = choice # Update array with move location
      for i in range(5):
        for j in range(5):
          print(moveset_left[i][j], end = " ")
        print("\t", end = " ")
        for j in range(5):
          print(moveset_right[i][j], end = " ")
        print("\t")
      print("\n")
  
    else:
      print("opponent (blue) moves available:") # Print opponent's info
      print()

      left_card = self.blue_cards[0]
      right_card = self.blue_cards[1]
      print(left_card + ":" + " "*(16 - len(left_card)) + right_card + ":") # Print card name 

      moveset_left = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_left[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[left_card]: # Loop between choices 
        i_change, j_change = self.cards[left_card][choice]
        j_change *= -1  # Account for player's reference frame 
        moveset_left[2+i_change][2+j_change] = choice # Update array with move location

      moveset_right = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset_right[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[right_card]: # Loop between choices 
        i_change, j_change = self.cards[right_card][choice]
        j_change *= -1  # Account for player's reference frame 
        moveset_right[2+i_change][2+j_change] = choice # Update array with move location
      for i in range(5):
        for j in range(5):
          print(moveset_left[i][j], end = " ")
        print("\t", end = ' ')
        for j in range(5):
          print(moveset_right[i][j], end = " ")
        print("\t")
      print("\n")

  @property
  def show_side_moves(self):
    
    """ Shows the moves for side card """
    
    print("side move:") # Print side move info
    print()
    for card in self.side_card: # Get side card
      print(card + ":") # Print card name 
      moveset = [['◦' for x in range(5)] for y in range(5)] # Initialise empty array 
      moveset[2][2] = '☺' # Place reference pawn piece 
      for choice in self.cards[card]: # Loop between choices 
        i_change, j_change = self.cards[card][choice]

        if self.whose_turn == 'blue':
          j_change *= -1  # Account for current turn player's reference frame 
        else:
          i_change *= -1 

        moveset[2+i_change][2+j_change] = choice # Update array with move location
      for row in moveset:
        for elem in row:
          print(elem, end = " ") # Print board state neatly
        print()
      print()
    
  @property
  def show_game_state(self):

    """ Shows current board, current player and opponent's moves and side cards moves """

    print("Turn " + str(self.number_of_turns))
    print()
    self.show_board
    self.show_player_moves
    self.show_opponent_moves
    self.show_side_moves
  
  def prompt_player_move(self, stage, quick_prompt = True):

    """ Prompt for decision from current turn player """
    # Stages are utilised to see degree of information provided by player
    # Stage 0: First time prompting
    # Stage 1: Choosing pawn onwards (reselecting pawn)
    # Stage 2: Choosing card onwards (reselecting card)
    # Stage 3: Choosing move onwards (reselecting move)
    # Stage 4: Completed selection

    # quick_prompt allows the user to input the piece, card & move in one line
    if quick_prompt:
      quick_input = input("Select your piece, card & move: ").split(' ') # A list of 3 items
      print(quick_input)

      # Deploy undo check
      if quick_input[0] == 'undo':
        self.undo_turn = eval(input("What turn do you want to go back to: "))
        if self.undo_turn > len(self.number_of_turns_log) - 1:
          print("undo turn is is invalid")
          return 0
        else:
          return "undo"

      try:
        # extract piece
        self.selected_pawn = quick_input[0]
        if '1' in self.selected_pawn or '2' in self.selected_pawn or '3' in self.selected_pawn or '4' in self.selected_pawn:
          self.selected_pawn = self.selected_pawn.lower()
        # extract card
        self.selected_card = quick_input[1].lower()
        # extract move
        self.selected_move = quick_input[2].upper()
      except:
        print("entered invalid string")
        return 0

      # Deploy all 8 checks for the three inputs
      if self.selected_pawn not in list(self.piece_state.keys()):
        print("piece is invalid. please reselect piece.")
        return 0
      elif self.whose_turn == 'blue' and 'r' in self.selected_pawn.lower():
        print("The blue player can only choose a blue piece. please reselect piece.")
        return 0
      elif self.whose_turn == 'red' and 'b' in self.selected_pawn.lower():
        print("The red player can only choose a red piece. please reselect piece.")
        return 0
      elif self.piece_state[self.selected_pawn] == -1:
        print("piece {} is dead, please reselect piece".format(self.selected_pawn))
        return 0
      elif (self.whose_turn == 'blue' and self.selected_card not in self.blue_cards) or (self.whose_turn == 'red' and self.selected_card not in self.red_cards):
        print("card is invalid. please reselect card.")
        return 0
      elif self.selected_move not in list(self.cards[self.selected_card].keys()):
        print("move is invalid. please reselect move.")
        return 0
      elif self.is_move_valid(mode = 'pvp') == False:
        return 0
      elif self.selected_pawn == None or self.selected_card == None or self.selected_move == None:
        print("error in selection. please reselect again")
        return 0
      else:
        return 4
    # Slower line by line prompt
    else:
      # print current player's turn 
      if stage == 0:
        print(self.whose_turn + "'s turn") 

      # prompt for pawn selection
      if stage == 0 or stage == 1:   
        self.selected_pawn = input("Select your piece (Enter 0 to reselect, undo to Undo): ")

        if '1' in self.selected_pawn or '2' in self.selected_pawn or '3' in self.selected_pawn or '4' in self.selected_pawn:
          self.selected_pawn = self.selected_pawn.lower()
        
        # undo function
        if self.selected_pawn == 'undo':
          self.undo_turn = eval(input("What turn do you want to go back to: "))
          if self.undo_turn > len(self.number_of_turns_log) - 1:
            print("undo turn is is invalid")
            return 1
          else:
            return "undo"  
        
        # account for reselection and ensure that selected_pawn is valid
        if self.selected_pawn == "0": 
          self.selected_pawn = None # standardise all selections back to None
          self.selected_card = None
          self.selected_move = None
          return 0 
        elif self.selected_pawn not in list(self.piece_state.keys()):
          print("piece is invalid. please reselect piece.")
          return 1
        elif self.whose_turn == 'blue' and 'r' in self.selected_pawn.lower():
          print("The blue player can only choose a blue piece. please reselect piece.")
          return 1
        elif self.whose_turn == 'red' and 'b' in self.selected_pawn.lower():
          print("The red player can only choose a red piece. please reselect piece.")
          return 1
        elif self.piece_state[self.selected_pawn] == -1:
          print("piece {} is dead, please reselect piece".format(self.selected_pawn))
          return 1
      
      # prompt for card selection
      if stage == 0 or stage == 1 or stage == 2:
        self.selected_card = input("Select your card (Enter 0 to reselect, undo to Undo): ").lower()
        
        # undo function
        if self.selected_card == 'undo':
          self.undo_turn = eval(input("What turn do you want to go back to: "))
          if self.undo_turn > len(self.number_of_turns_log) - 1:
            print("undo turn is is invalid")
            return 1
          else:
            return "undo"  

        # ensure that card is valid
        if self.selected_card == "0":
          self.selected_pawn = None
          self.selected_card = None
          self.selected_move = None
          return 0 
        elif (self.whose_turn == 'blue' and self.selected_card not in self.blue_cards) or (self.whose_turn == 'red' and self.selected_card not in self.red_cards):
          print("card is invalid. please reselect card.")
          return 2
      
      # prompt for move selection
      if stage == 0 or stage == 1 or stage == 2 or stage == 3: 
        self.selected_move = input("Select your move (Enter 0 to reselect, undo to Undo): ").upper()
        
        # undo function
        if self.selected_move == 'undo':
          self.undo_turn = eval(input("What turn do you want to go back to: "))
          if self.undo_turn > len(self.number_of_turns_log) - 1:
            print("undo turn is is invalid")
            return 1
          else:
            return "undo"  

        # ensure that move is valid
        if self.selected_card == "0":
          self.selected_pawn = None
          self.selected_card = None
          self.selected_move = None
          return 0 
        elif self.selected_move not in list(self.cards[self.selected_card].keys()):
          print("move is invalid. please reselect move.")
          return 3
        elif self.is_move_valid(mode = 'pvp') == False:
          return 3
      
      # raise error if there is empty selection and redirect to reselection
      if self.selected_pawn == None or self.selected_card == None or self.selected_move == None:
        print("error in selection. please reselect again")
        return 0
      else:
        return 4

  def is_move_valid(self, mode):

    """ this function check if selected move is valid given board state """
    
    # Note that the first move in the card is called A, followed by B, C & D for up to 4 moves
    # self.selected_move (input into this function (not directly, but as a class attribute)) refers to this A, B, C or D

    # determine the coordinate change of the piece and current coordinate of the piece
    i_change, j_change = self.cards[self.selected_card][self.selected_move]
    i_current = self.piece_state[self.selected_pawn][0]
    j_current = self.piece_state[self.selected_pawn][1]

    # Transpose coordinate change based on the player. This is hardcoded to the coordinate system of the board
    if self.whose_turn == 'blue':
      j_change *= -1
    else:
      i_change *= -1

    # ensure that new coordinate (current + change) are valid on the board 
    # in other words it cannot be outside the board and it cannot be clashing with a piece from the SAME team
    i_new = i_current + i_change
    j_new = j_current + j_change

    if i_new < 0 or i_new > 4 or j_new < 0 or j_new > 4: # outside the board
      if mode == 'pvp':
        print("move is invalid as the piece falls outside the board. please reselect move")
      return False
    if self.whose_turn == 'blue':
      if 'b' in self.board[i_new][j_new].lower():
        if mode == 'pvp':
          print("a blue piece cannot take the place of another blue piece. please reselect move")
        return False
    else:
      if 'r' in self.board[i_new][j_new].lower():
        if mode == 'pvp':
          print("a red piece cannot take the place of another red piece. please reselect move")
        return False

  def are_ai_selections_valid(self): # a copy of the prompt_player_move method for AI

    """ check the validity of the moves selected by the ai  """

    # check for validity of pawn
    if self.selected_pawn not in list(self.piece_state.keys()):
      return False
    elif self.piece_state[self.selected_pawn] == -1:
      return False
    elif self.whose_turn == 'blue' and 'r' in self.selected_pawn.lower():
      return False
    elif self.whose_turn == 'red' and 'b' in self.selected_pawn.lower():
      return False

    # check for validity of card
    if (self.whose_turn == 'blue' and self.selected_card not in self.blue_cards) or (self.whose_turn == 'red' and self.selected_card not in self.red_cards):
      return False
    
    # check for validity of move
    if self.selected_move not in list(self.cards[self.selected_card].keys()):
      return False
    elif self.is_move_valid(mode = 'ai') == False:
      return False
    
    # raise error if there is empty selection and redirect to reselection
    if self.selected_pawn == None or self.selected_card == None or self.selected_move == None:
      return False
    else:
      return True

  def move(self):

    """ This method implements a move and updates the game state accordingly """

    # Note that the first move in the card is called A, followed by B, C & D for up to 4 moves
    # self.selected_move (input into this function) refers to this A, B, C or D

    # determine the coordinate change of the piece and current coordinate of the piece
    i_change, j_change = self.cards[self.selected_card][self.selected_move]
    i_current = self.piece_state[self.selected_pawn][0]
    j_current = self.piece_state[self.selected_pawn][1]

    # Transpose coordinate change based on the player. This is hardcoded to the coordinate system of the board
    if self.whose_turn == 'blue':
      j_change *= -1
    else:
      i_change *= -1

    # ensure that new coordinate (current + change) are valid on the board 
    # in other words it cannot be outside the board and it cannot be clashing with a piece from the SAME team
    i_new = i_current + i_change
    j_new = j_current + j_change

    # first we identify if the move kills a piece from the opposing team

    if self.whose_turn == 'blue':
      if 'r' in self.board[i_new][j_new].lower():
        dead_piece = self.board[i_new][j_new]
        self.piece_state[dead_piece] = -1 # mark red piece as dead
    else:
      if 'b' in self.board[i_new][j_new].lower():
        dead_piece = self.board[i_new][j_new]
        self.piece_state[dead_piece] = -1 # mark blue piece as dead

    # now we update the piece state of the piece into its new position
    self.piece_state[self.selected_pawn] = [i_new, j_new]

    # afterwards, we update the board
    self.board[i_current][j_current] = '•'
    self.board[i_new][j_new] = self.selected_pawn
    
    return True

  def turn_pvp(self):
    
    """ This method implements the process of one turn of gameplay for pvp"""

    # show game_state
    self.show_game_state

    # prompt player to make decision
    stage = 0 
    while stage != 4:
      stage = self.prompt_player_move(stage)
      if stage == "undo":
        self.load_history(self.undo_turn) # load to desired turn 
        return

    # move the pieces and update the board
    self.move()

    # update the blue_win or red_win booleans if win condition is met
    self.check_win()

    # now update the position of the cards, the player of this turn will get the side card and discard used card to the side
    if self.whose_turn == 'blue':
      self.blue_cards.append(self.side_card[0])
      self.blue_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]
    else:
      self.red_cards.append(self.side_card[0])
      self.red_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]

    # finally, change turns
    if self.whose_turn == 'blue':
      self.whose_turn = 'red'
    else:
      self.whose_turn = 'blue'

    # save current game state
    self.save_history()
    # increment the number of turns
    self.number_of_turns += 1

  def turn_ai(self):
    
    """ This method implements the process of one turn of gameplay for ai (presumes moves have already been chosen prior and are valid)"""
    
    # move the pieces and update the board
    self.move()

    # update the blue_win or red_win booleans if win condition is met
    self.check_win()

    # now update the position of the cards, the player of this turn will get the side card and discard used card to the side
    if self.whose_turn == 'blue':
      self.blue_cards.append(self.side_card[0])
      self.blue_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]
    else:
      self.red_cards.append(self.side_card[0])
      self.red_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]

    # finally, change turns
    if self.whose_turn == 'blue':
      self.whose_turn = 'red'
    else:
      self.whose_turn = 'blue'

    # save current game state
    self.save_history()
    # increment the number of turns
    self.number_of_turns += 1

    return True

  def turn_minimax(self, minimax_depth = None, parallel = None, prune = True, return_move_index = False):
    
    """ This method implements the process of one turn of gameplay for minimax algo"""
    self.prune = prune

    # manual setting of depth in minimax tree
    if minimax_depth != None:
      self.depth = minimax_depth
    
    # show game_state
    if self.verbose:
      self.show_game_state

    # call minimax algo and choose selected move given current state
    if parallel == "Process":
      value = self.multi_minimax_Process(depth = 0, alpha = -math.inf, beta = math.inf)
    if parallel == "Pool":
      value = self.multi_minimax_Pool(depth = 0, alpha = -math.inf, beta = math.inf)
    else:
      value = self.minimax(depth = 0, alpha = -math.inf, beta = math.inf, return_move_index = return_move_index)
    if self.verbose:
      print("Turn {}: ".format(self.number_of_turns),self.best_selected_pawn, self.best_selected_card, self.best_selected_move)

    self.selected_pawn = self.best_selected_pawn
    self.selected_card = self.best_selected_card
    self.selected_move = self.best_selected_move

    # move the pieces and update the board
    self.move()
    
    # update the blue_win or red_win booleans if win condition is met
    self.check_win()

    # now update the position of the cards, the player of this turn will get the side card and discard used card to the side
    if self.whose_turn == 'blue':
      self.blue_cards.append(self.side_card[0])
      self.blue_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]
    else:
      self.red_cards.append(self.side_card[0])
      self.red_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]

    # finally, change turns
    if self.whose_turn == 'blue':
      self.whose_turn = 'red'
    else:
      self.whose_turn = 'blue'

    # save history
    self.save_history()
    # increment the number of turns
    self.number_of_turns += 1

  def turn_mcts(self, mcts_object):

    """ This method implements the process of one turn of gameplay for monte-carlo search tree algo"""

    # show game_state
    if self.verbose:
      self.show_game_state

    # execute MCTS and obtain tuple of actions (piece, card, move)
    action_tuple = mcts_object.search(initialState = self)

    if self.verbose:
      print("Turn {}: ".format(self.number_of_turns),action_tuple[0], action_tuple[1], action_tuple[2])

    self.selected_pawn = action_tuple[0]
    self.selected_card = action_tuple[1]
    self.selected_move = action_tuple[2]

    # move the pieces and update the board
    self.move()

    # update the blue_win or red_win booleans if win condition is met
    self.check_win()

    # now update the position of the cards, the player of this turn will get the side card and discard used card to the side
    if self.whose_turn == 'blue':
      self.blue_cards.append(self.side_card[0])
      self.blue_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]
    else:
      self.red_cards.append(self.side_card[0])
      self.red_cards.remove(self.selected_card)
      self.side_card = [self.selected_card]

    # finally, change turns
    if self.whose_turn == 'blue':
      self.whose_turn = 'red'
    else:
      self.whose_turn = 'blue'

    # need to save history first
    self.save_history()
    # increment the number of turns
    self.number_of_turns += 1

  def check_win(self):

    """ This method checks if any player has won and updates self.blue_win or self.red_win if necessary """

    # check if blue wins (either R is no longer on the board or there is a blue master at (4,2))
    if self.piece_state['R'] == -1 or 'B' in self.board[4][2]:
      self.blue_win = True

    # check if red wins (either B is no longer on the board or there is a red master at (0,2))
    if self.piece_state['B'] == -1 or 'R' in self.board[0][2]:
      self.red_win = True

    # ensure both teams cannot win at the same time (if so, there is an error in the onitama class source code)
    if self.blue_win and self.red_win:
      raise ValueError("both blue and red have won, check the source code for error(s)")

  def save_history(self):
    
    """ this function saves every game variable for every turn """
    
    # if past entry for turn exists in same index, replace it
    if self.number_of_turns == len(self.number_of_turns_log) - 1:
      self.whose_turn_log[self.number_of_turns] = self.whose_turn
      self.number_of_turns_log[self.number_of_turns] = self.number_of_turns
      self.blue_win_log[self.number_of_turns] = self.blue_win
      self.red_win_log[self.number_of_turns] = self.red_win
      self.blue_cards_log[self.number_of_turns] = self.blue_cards[:]
      self.red_cards_log[self.number_of_turns] = self.red_cards[:]
      self.side_card_log[self.number_of_turns] = self.side_card[:]
      self.board_log[self.number_of_turns] = [x[:] for x in self.board]
      self.piece_state_log[self.number_of_turns] = self.piece_state.copy()
      self.selected_pawn_log[self.number_of_turns] = self.selected_pawn
      self.selected_card_log[self.number_of_turns] = self.selected_card
      self.selected_move_log[self.number_of_turns] = self.selected_move
    # else append new entry for new turn
    else:
      self.whose_turn_log.append(self.whose_turn)
      self.number_of_turns_log.append(self.number_of_turns)
      self.blue_win_log.append(self.blue_win)
      self.red_win_log.append(self.red_win)
      self.blue_cards_log.append(self.blue_cards[:])
      self.red_cards_log.append(self.red_cards[:])
      self.side_card_log.append(self.side_card[:])
      self.board_log.append([x[:] for x in self.board])
      self.piece_state_log.append(self.piece_state.copy())
      self.selected_pawn_log.append(self.selected_pawn) 
      self.selected_card_log.append(self.selected_card)
      self.selected_move_log.append(self.selected_move)

  def load_history(self, turn_num, remove_obsolete_history = True): 

    """ this functions reinstates all the game variables for a specified turn BEFORE MOVE ON TURN IS MADE """

    self.whose_turn = self.whose_turn_log[turn_num - 1]
    self.number_of_turns = self.number_of_turns_log[turn_num - 1]
    self.blue_win = self.blue_win_log[turn_num - 1]
    self.red_win = self.red_win_log[turn_num - 1]
    self.blue_cards = self.blue_cards_log[turn_num - 1][:]
    self.red_cards = self.red_cards_log[turn_num - 1][:]
    self.side_card = self.side_card_log[turn_num - 1][:]
    self.board = [x[:] for x in self.board_log[turn_num - 1]]
    self.piece_state = self.piece_state_log[turn_num - 1].copy()
    self.selected_pawn = self.selected_pawn_log[turn_num - 1]
    self.selected_card = self.selected_card_log[turn_num - 1]
    self.selected_move = self.selected_move_log[turn_num - 1]

    # set number_of_turns accordingly
    self.number_of_turns = turn_num

    if remove_obsolete_history:
      # remove obsolete history
      self.remove_obsolete_history()

  def remove_obsolete_history(self):

    """ this function removes redundant entries in log """

    self.whose_turn_log =  self.whose_turn_log[:self.number_of_turns]
    self.number_of_turns_log =  self.number_of_turns_log[:self.number_of_turns]
    self.blue_win_log =  self.blue_win_log[:self.number_of_turns]
    self.red_win_log =  self.red_win_log[:self.number_of_turns]
    self.blue_cards_log=  self.blue_cards_log[:self.number_of_turns]
    self.red_cards_log =  self.red_cards_log[:self.number_of_turns]
    self.side_card_log =  self.side_card_log[:self.number_of_turns]
    self.board_log =  self.board_log[:self.number_of_turns]
    self.piece_state_log =  self.piece_state_log[:self.number_of_turns]
    self.selected_pawn_log =  self.selected_pawn_log[:self.number_of_turns]
    self.selected_card_log =  self.selected_card_log[:self.number_of_turns]
    self.selected_move_log =  self.selected_move_log[:self.number_of_turns]

  def eval_board_state(self):

    """ This function evaluates the board state based on OUR preconceived notion of value """

    # initialise score
    score = 0

    # win condition game state
    '''
    if self.red_win == True:
      score += 5000
      return score
    elif self.blue_win == True:
      score -= 5000
      return score
    '''
    
    # evaluates the piece state. assume red as maximising player and blue as minimizing
    for piece in list(self.piece_state.keys()):
      if 'r' in piece and self.piece_state[piece] != -1:
        score += 10 # add score of 10 for red student pawn 
      elif 'R' in piece and self.piece_state[piece] != -1:
        score += 150 # add score of 100 for red master pawn
        i_master = self.piece_state[piece][0]
        j_master = self.piece_state[piece][1]
        dist_score = 3*pow(pow(i_master - 0, 2) + pow(j_master - 2, 2), 0.5) # takes euclidian distance to 2nd win condition as score
        score -= dist_score # adds -distance score for red     
      elif 'b' in piece and self.piece_state[piece] != -1:
        score = score - 10 # add score of -10 for blue student pawn
      elif 'B' in piece and self.piece_state[piece] != -1:
        score -= 150 # add score of -100 for blue master pawn
        i_master = self.piece_state[piece][0]
        j_master = self.piece_state[piece][1]
        dist_score = 3*pow(pow(i_master - 4, 2) + pow(j_master - 2, 2), 0.5) # takes euclidian distance to 2nd win condition as score
        score += dist_score # adds distance score for blue

    return score

  def minimax(self, depth, alpha, beta, return_move_index = False):

    """ this function implement minimax with alpha - beta pruning """
    """ algo should return the original game state and best move, given original game state, stored in the best_selected variables """

    if depth == self.depth or self.blue_win == True or self.red_win == True:
      eval = self.eval_board_state()
      return eval

    # to keep track of move_index for Deep RL training (start with -1 as first increment brings index to 0, the first valid index)
    if return_move_index and depth == 0:
      move_index = -1

    if self.whose_turn == 'red':
      # create temp lists for red
      temp_red_pieces = self.red_pieces[:]
      temp_red_cards = self.red_cards[:]
      max_score = -math.inf
      for piece in temp_red_pieces: # loop through pieces
        for card in temp_red_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if return_move_index and depth == 0:
              move_index += 1
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              self.turn_ai() # execute moves on board 

            score = self.minimax(depth + 1, alpha, beta) # recursively call minimax for score of next node
            if score > max_score:
              max_score = score
              if depth == 0:
                # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
                self.best_selected_pawn = self.selected_pawn_log[self.number_of_turns - 1] # saves best pawn selection
                self.best_selected_card = self.selected_card_log[self.number_of_turns - 1] # saves best card selection
                self.best_selected_move = self.selected_move_log[self.number_of_turns - 1] # saves best move selection
                # save best move_index if needed
                if return_move_index:
                  self.selected_move_index = move_index 
            if self.prune:
              if max_score >= beta: 
                self.load_history(self.number_of_turns - 1)
                return max_score
              if max_score > alpha:
                alpha = max_score
            self.load_history(self.number_of_turns - 1) # loads the orignal game state (previous node)
      return max_score

      
    elif self.whose_turn == 'blue':
      temp_blue_pieces = self.blue_pieces[:]
      temp_blue_cards = self.blue_cards[:]
      min_score = math.inf
      for piece in temp_blue_pieces: # loop through pieces
        for card in temp_blue_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if return_move_index and depth == 0:
              move_index += 1
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              self.turn_ai() # execute moves on board 

            score = self.minimax(depth + 1, alpha, beta) # recursively call minimax for score of next node
            if score < min_score:
              min_score = score
              if depth == 0:
                # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
                self.best_selected_pawn = self.selected_pawn_log[self.number_of_turns - 1] # saves best pawn selection
                self.best_selected_card = self.selected_card_log[self.number_of_turns - 1] # saves best card selection
                self.best_selected_move = self.selected_move_log[self.number_of_turns - 1] # saves best move selection
                # save best move_index if needed
                if return_move_index:
                  self.selected_move_index = move_index 
            if self.prune:
              if min_score <= alpha: 
                self.load_history(self.number_of_turns - 1)
                return min_score
              if min_score < beta:
                beta = min_score
            self.load_history(self.number_of_turns - 1) # resets the game state (previous node)
      return min_score

  def dummy_minimax(self, action = None, depth = 1, alpha = -math.inf, beta = math.inf, scores = None, count = None, parallel_method = "Pool"):

    """ this function implement minimax with alpha - beta pruning """
    """ algo should return the original game state and best move, given original game state, stored in the best_selected variables """

    if depth == 1:
      self.selected_pawn = action[0]
      self.selected_card = action[1]
      self.selected_move = action[2]
      self.turn_ai() 

    if depth == self.depth or self.blue_win == True or self.red_win == True:
      eval = self.eval_board_state()
      if depth == 1:
        if parallel_method == "Process":
          scores[count] = eval
        else:
          return eval
      else:
        return eval

    if self.whose_turn == 'red':
      # create temp lists for red
      temp_red_pieces = self.red_pieces[:]
      temp_red_cards = self.red_cards[:]
      max_score = -math.inf
      for piece in temp_red_pieces: # loop through pieces
        for card in temp_red_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              self.turn_ai() # execute moves on board 

            score = self.dummy_minimax(depth = depth + 1, alpha = alpha, beta = beta) # recursively call minimax for score of next node
            if score > max_score:
              max_score = score
            if self.prune:
              if max_score >= beta: 
                self.load_history(self.number_of_turns - 1)
                if depth == 1:
                  if parallel_method == "Process":
                    scores[count] = max_score
                  else:
                    return max_score
                else:
                  return max_score
              if max_score > alpha:
                alpha = max_score
            self.load_history(self.number_of_turns - 1) # loads the orignal game state (previous node)
      if depth == 1:
        if parallel_method == "Process":
          scores[count] = max_score
        else:
          return max_score
      else:
        return max_score
      
    elif self.whose_turn == 'blue':
      temp_blue_pieces = self.blue_pieces[:]
      temp_blue_cards = self.blue_cards[:]
      min_score = math.inf
      for piece in temp_blue_pieces: # loop through pieces
        for card in temp_blue_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              self.turn_ai() # execute moves on board 

            score = self.dummy_minimax(depth = depth + 1, alpha = alpha, beta = beta) # recursively call minimax for score of next node
            if score < min_score:
              min_score = score
            if self.prune:
              if min_score <= alpha: 
                self.load_history(self.number_of_turns - 1)
                if depth == 1:
                  if parallel_method == "Process":
                    scores[count] = min_score
                  else:
                    return min_score
                else:
                  return min_score
              if min_score < beta:
                beta = min_score
            self.load_history(self.number_of_turns - 1) # resets the game state (previous node)
      if depth == 1:
        if parallel_method == "Process":
          scores[count] = min_score
        else:
          return min_score
      else:
        return min_score

  def multi_minimax_Process(self, depth, alpha, beta):

    """ this function implement minimax with alpha - beta pruning """
    """ algo should return the original game state and best move, given original game state, stored in the best_selected variables """
    """ This function makes use of multiprocessing to speed up the minimax algorithm """

    number_of_cpus = mp.cpu_count()

    if self.whose_turn == 'red':
      # create temp lists for red
      actions = []
      temp_red_pieces = self.red_pieces[:]
      temp_red_cards = self.red_cards[:]
      max_score = -math.inf
      for piece in temp_red_pieces: # loop through pieces
        for card in temp_red_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              actions.append((piece, card, move))

      procs = []
      scores = mp.Array('d', len(actions))
      for count, action in enumerate(actions):
        proc = mp.Process(target = self.dummy_minimax, args = (action, depth + 1, alpha, beta, scores, count, "Process"))
        # if len(procs) == number_of_cpus:
        #   procs[0].join()
        #   del procs[0]
        while len(procs) == number_of_cpus:
          for i, p in enumerate(procs):
            if not p.is_alive():
              del procs[i]
              break
        procs.append(proc)
        proc.start()

      for proc in procs:
        proc.join()

      for count, action in enumerate(actions):
        if scores[count] > max_score:
          max_score = scores[count]
          # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
          self.best_selected_pawn = action[0] # saves best pawn selection
          self.best_selected_card = action[1] # saves best card selection
          self.best_selected_move = action[2] # saves best move selection 
        if max_score >= beta: 
          # self_copy.load_history(self_copy.number_of_turns - 1)
          return max_score
        if max_score > alpha:
          alpha = max_score
        # self_copy.load_history(self_copy.number_of_turns - 1) # loads the orignal game state (previous node)
      return max_score

      
    elif self.whose_turn == 'blue':
      actions = []
      temp_blue_pieces = self.blue_pieces[:]
      temp_blue_cards = self.blue_cards[:]
      min_score = math.inf
      for piece in temp_blue_pieces: # loop through pieces
        for card in temp_blue_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              actions.append((piece, card, move))

      procs = []
      scores = mp.Array('d', len(actions))
      for count, action in enumerate(actions):
        proc = mp.Process(target = self.dummy_minimax, args = (action, depth + 1, alpha, beta, scores, count, "Process"))
        # if len(procs) == number_of_cpus:
        #   procs[0].join()
        #   del procs[0]
        while len(procs) == number_of_cpus:
          for i, p in enumerate(procs):
            if not p.is_alive():
              del procs[i]
              break        
        procs.append(proc)
        proc.start()

      for proc in procs:
        proc.join()
      
      for count, action in enumerate(actions):
        if scores[count] < min_score:
          min_score = scores[count]
          # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
          self.best_selected_pawn = action[0] # saves best pawn selection
          self.best_selected_card = action[1] # saves best card selection
          self.best_selected_move = action[2] # saves best move selection 
        if min_score <= alpha: 
          # self_copy.load_history(self_copy.number_of_turns - 1)
          return min_score
        if min_score < beta:
          beta = min_score
        # self_copy.load_history(self_copy.number_of_turns - 1) # loads the orignal game state (previous node)
      return min_score

  def multi_minimax_Pool(self, depth, alpha, beta):

    """ this function implement minimax with alpha - beta pruning """
    """ algo should return the original game state and best move, given original game state, stored in the best_selected variables """
    """ This function makes use of multiprocessing to speed up the minimax algorithm """

    number_of_cpus = mp.cpu_count()

    if self.whose_turn == 'red':
      # create temp lists for red
      actions = []
      temp_red_pieces = self.red_pieces[:]
      temp_red_cards = self.red_cards[:]
      max_score = -math.inf
      for piece in temp_red_pieces: # loop through pieces
        for card in temp_red_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              actions.append((piece, card, move))

      pool = mp.Pool(len(actions))
      results = pool.map(self.dummy_minimax, actions)

      for count, action in enumerate(actions):
        if results[count] > max_score:
          max_score = results[count]
          # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
          self.best_selected_pawn = action[0] # saves best pawn selection
          self.best_selected_card = action[1] # saves best card selection
          self.best_selected_move = action[2] # saves best move selection 
        if max_score >= beta: 
          # self_copy.load_history(self_copy.number_of_turns - 1)
          return max_score
        if max_score > alpha:
          alpha = max_score
        # self_copy.load_history(self_copy.number_of_turns - 1) # loads the orignal game state (previous node)
      return max_score

      
    elif self.whose_turn == 'blue':
      actions = []
      temp_blue_pieces = self.blue_pieces[:]
      temp_blue_cards = self.blue_cards[:]
      min_score = math.inf
      for piece in temp_blue_pieces: # loop through pieces
        for card in temp_blue_cards: # loop through cards
          for move in list(self.cards[card].keys()): # loop through moves
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              continue # if selections are not valid, move to next iteration
            else:
              actions.append((piece, card, move))

      pool = mp.Pool(len(actions))
      results = pool.map(self.dummy_minimax, actions)
      
      for count, action in enumerate(actions):
        if results[count] < min_score:
          min_score = results[count]
          # The -1 is needed in the index as the number of turns variable has already incremented, and we care about the previous turn selections
          self.best_selected_pawn = action[0] # saves best pawn selection
          self.best_selected_card = action[1] # saves best card selection
          self.best_selected_move = action[2] # saves best move selection 
        if min_score <= alpha: 
          # self_copy.load_history(self_copy.number_of_turns - 1)
          return min_score
        if min_score < beta:
          beta = min_score
        # self_copy.load_history(self_copy.number_of_turns - 1) # loads the orignal game state (previous node)
      return min_score

      """ The following methods are used by the MCTS class"""

  def getCurrentPlayer(self):

    """ Returns 1 for maximising player (red) and -1 for minimising_player (blue) """

    if self.whose_turn == "red":
      return 1
    else:
      return -1

  def getPossibleActions(self, valid = True):

    """ Returns a list of actions with each action being a tuple in the form (piece, card, move) """

    # empty list of actions created
    actions = []

    if self.whose_turn == "red":

      for piece in self.red_pieces:
        for card in self.red_cards:
          for move in list(self.cards[card].keys()):
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              # if valid is true, then only the valid actions are counted
              if valid:
                continue
            actions.append((piece,card,move))
    else: # blue player

      for piece in self.blue_pieces:
        for card in self.blue_cards:
          for move in list(self.cards[card].keys()):
            self.selected_pawn = piece
            self.selected_card = card
            self.selected_move = move
            if self.are_ai_selections_valid() == False:
              # if valid is true, then only the valid actions are counted
              if valid:
                continue
            actions.append((piece,card,move))

    # may not be necessary, but we set the selected varaibles back to None
    self.selected_pawn = None
    self.selected_card = None
    self.selected_move = None

    return actions

  def isTerminal(self):

    """ True if game has ended """

    if self.blue_win == True or self.red_win == True:
      return True
    else:
      return False

  def getReward(self):

    """ Returns the game state evaluation reward for MCTS after terminal state is hit """

    if self.blue_win:
      return -1
    elif self.red_win:
      return 1

  def takeAction(self, action):

    """" Implements the specified action to the board game """

    self.selected_pawn = action[0]
    self.selected_card = action[1]
    self.selected_move = action[2]

    self.turn_ai()

    return self

  def replay_game(self):

    """ Prints out an entire replay of the game """

    turns_needed = len(self.number_of_turns_log)

    for turn in range(turns_needed):
      self.load_history(turn + 1, remove_obsolete_history = False)
      if turn != 0:
        print("Turn {} Move Made: ".format(self.number_of_turns - 1), self.selected_pawn, self.selected_card, self.selected_move)
      self.show_game_state



  def return_NN_state(self, perspective = None, use_hardcoded = False):

    """ Returns the board game state information in a 1x4x5x5 array (Board State) 
    and a 1x48 array (Game Card State) as input into a Neural Network for Deep RL
    for the current player's turn """

    # initialise the two arrays to zero
    # board state array of shape (4,5,5) --> channel, i, j
    board_state = np.zeros([1,4,5,5])
    # card state array of shape (1,48) --> 16 own card, 16 side card, 16 opponent card
    if use_hardcoded:
      card_state = np.zeros([1,10])
    else:
      card_state = np.zeros([1,48])

    if perspective == 'red' or (perspective == None and self.whose_turn == "red"):

      # first gather the 4x5x5 board state data 
      # First channel is master piece of current player (1 represents location of master piece while 0 does not)
      if self.piece_state['R'] != -1:
        board_state[0][0][self.piece_state['R'][0]][self.piece_state['R'][1]] = 1

      # Second channel has information on pawns of current player
      for i in range(4): # loop through the 4 pawns
        piece = 'r'+ str(i+1)
        if self.piece_state[piece] != -1: # ensure piece is not dead
          board_state[0][1][self.piece_state[piece][0]][self.piece_state[piece][1]] = 1

      # Third channel is the master piece of the opponent
      if self.piece_state['B'] != -1:
        board_state[0][2][self.piece_state['B'][0]][self.piece_state['B'][1]] = 1

      # Fourth channel has information on pawns of opponent player
      for i in range(4): # loop through the 4 pawns
        piece = 'b'+ str(i+1)
        if self.piece_state[piece] != -1: # ensure piece is not dead
          board_state[0][3][self.piece_state[piece][0]][self.piece_state[piece][1]] = 1

      if use_hardcoded:
        card_state[0][list(self.hardcoded_cards.keys()).index(self.red_cards[0])] = 1
        card_state[0][list(self.hardcoded_cards.keys()).index(self.red_cards[1])] = 1
        # First identify cards of current player
        card_state[0][list(self.hardcoded_cards.keys()).index(self.blue_cards[0]) + 5] = 1
        card_state[0][list(self.hardcoded_cards.keys()).index(self.blue_cards[1]) + 5] = 1
      else:
        # now, gather information on the card state data
        # First identify cards of current player
        card_state[0][list(self.card_colour.keys()).index(self.red_cards[0])] = 1
        card_state[0][list(self.card_colour.keys()).index(self.red_cards[1])] = 1
        # Second identify side card
        card_state[0][list(self.card_colour.keys()).index(self.side_card[0]) + 16] = 1
        # First identify cards of current player
        card_state[0][list(self.card_colour.keys()).index(self.blue_cards[0]) + 32] = 1
        card_state[0][list(self.card_colour.keys()).index(self.blue_cards[1]) + 32] = 1

    else:

      # first gather the 4x5x5 board state data 
      # First channel is master piece of current player (1 represents location of master piece while 0 does not)
      if self.piece_state['B'] != -1:
        board_state[0][0][self.piece_state['B'][0]][self.piece_state['B'][1]] = 1

      # Second channel has information on pawns of current player
      for i in range(4): # loop through the 4 pawns
        piece = 'b'+ str(i+1)
        if self.piece_state[piece] != -1: # ensure piece is not dead
          board_state[0][1][self.piece_state[piece][0]][self.piece_state[piece][1]] = 1

      # Third channel is the master piece of the opponent
      if self.piece_state['R'] != -1:
        board_state[0][2][self.piece_state['R'][0]][self.piece_state['R'][1]] = 1

      # Fourth channel has information on pawns of opponent player
      for i in range(4): # loop through the 4 pawns
        piece = 'r'+ str(i+1)
        if self.piece_state[piece] != -1: # ensure piece is not dead
          board_state[0][3][self.piece_state[piece][0]][self.piece_state[piece][1]] = 1

      if use_hardcoded:
        card_state[0][list(self.hardcoded_cards.keys()).index(self.blue_cards[0])] = 1
        card_state[0][list(self.hardcoded_cards.keys()).index(self.blue_cards[1])] = 1
        # First identify cards of current player
        card_state[0][list(self.hardcoded_cards.keys()).index(self.red_cards[0]) + 5] = 1
        card_state[0][list(self.hardcoded_cards.keys()).index(self.red_cards[1]) + 5] = 1
      else:
        # now, gather information on the card state data
        # First identify cards of current player
        card_state[0][list(self.card_colour.keys()).index(self.blue_cards[0])] = 1
        card_state[0][list(self.card_colour.keys()).index(self.blue_cards[1])] = 1
        # Second identify side card
        card_state[0][list(self.card_colour.keys()).index(self.side_card[0]) + 16] = 1
        # First identify cards of current player
        card_state[0][list(self.card_colour.keys()).index(self.red_cards[0]) + 32] = 1
        card_state[0][list(self.card_colour.keys()).index(self.red_cards[1]) + 32] = 1

    # return both states
    return board_state, card_state
  
  def check_NN_valid_move(self, move_index):

    """ Takes in an integer from 0 to 39 (move_index) and checks if it is valid """
    # Note that every 8 index is a piece, in which each 4 is the card and then in each 4 we have 4 moves***

    # extract the selected piece, card and move (irregardless of its existence/validity)
    if self.whose_turn == "red":
      # find associated piece
      self.selected_pawn = self.red_pieces[int(move_index/8)]
      # find associated card (based on order) of which comes first in the list of cards
      if list(self.card_colour.keys()).index(self.red_cards[0]) < list(self.card_colour.keys()).index(self.red_cards[1]):
        if move_index%8 < 4:
          self.selected_card = self.red_cards[0]
        else:
          self.selected_card = self.red_cards[1]
      else:
        if move_index%8 < 4:
          self.selected_card = self.red_cards[1]
        else:
          self.selected_card = self.red_cards[0]
      # find associated move
      move_num = move_index % 4 # if move does not exist, return 0
      self.selected_move = ['A', 'B', 'C', 'D'][move_num]

    else:
      # find associated piece
      self.selected_pawn = self.blue_pieces[int(move_index/8)]
      # find associated card (based on order) of which comes first in the list of cards
      if list(self.card_colour.keys()).index(self.blue_cards[0]) < list(self.card_colour.keys()).index(self.blue_cards[1]):
        if move_index%8 < 4:
          self.selected_card = self.blue_cards[0]
        else:
          self.selected_card = self.blue_cards[1]
      else:
        if move_index%8 < 4:
          self.selected_card = self.blue_cards[1]
        else:
          self.selected_card = self.blue_cards[0]
      # find associated move
      move_num = move_index % 4 # if move does not exist, return 0
      self.selected_move = ['A', 'B', 'C', 'D'][move_num]

    if self.are_ai_selections_valid():
      return 1
    else:
      return 0

  def check_NN_valid_move_space(self):

    """ This method uses the check_NN_valid_move method to output a list of size 40 for valid actions """

    return [self.check_NN_valid_move(index) for index in range(40)]

  def turn_deeprl(self, move_index):
    
    """ This method implements the process of one turn of gameplay for deep rl algo based on a move_index (0-39)
    The move is assumed to already be valid """

    # extract the selected piece, card and move
    if self.whose_turn == "red":
      # find associated piece
      self.selected_pawn = self.red_pieces[int(move_index/8)]
      # find associated card (based on order) of which comes first in the list of cards
      if list(self.card_colour.keys()).index(self.red_cards[0]) < list(self.card_colour.keys()).index(self.red_cards[1]):
        if move_index%8 < 4:
          self.selected_card = self.red_cards[0]
        else:
          self.selected_card = self.red_cards[1]
      else:
        if move_index%8 < 4:
          self.selected_card = self.red_cards[1]
        else:
          self.selected_card = self.red_cards[0]
      # find associated move
      move_num = move_index % 4 # if move does not exist, return 0
      self.selected_move = ['A', 'B', 'C', 'D'][move_num]

    else:
      # find associated piece
      self.selected_pawn = self.blue_pieces[int(move_index/8)]
      # find associated card (based on order) of which comes first in the list of cards
      if list(self.card_colour.keys()).index(self.blue_cards[0]) < list(self.card_colour.keys()).index(self.blue_cards[1]):
        if move_index%8 < 4:
          self.selected_card = self.blue_cards[0]
        else:
          self.selected_card = self.blue_cards[1]
      else:
        if move_index%8 < 4:
          self.selected_card = self.blue_cards[1]
        else:
          self.selected_card = self.blue_cards[0]

    # show game_state
    if self.verbose:
      self.show_game_state

    # execute one turn
    self.turn_ai()