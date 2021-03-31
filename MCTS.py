from std_imports import *

# random policy function used for MCTS
def randomPolicy(state):
  while not state.isTerminal():
      try:
          action = random.choice(state.getPossibleActions())
      except IndexError:
          state.show_game_state
          raise Exception("Non-terminal state has no possible actions: " + str(state))
      state = state.takeAction(action)
  return state.getReward()

# Space Efficient representation of the Tree node class for MCTS
# Does not save the state, instead it uses reference values from action
# to play the game whenever the state is "set". Traversing a tree using
# such nodes is slow, but space efficient
class treeNode_spaceEfficient():
  def __init__(self, action = None, parent = None):
      self.action = action
      self.parent = parent
      self.numVisits = 0
      self.totalReward = 0
      self.children = {}
      self.num_blue_wins = 0
      self.num_red_wins = 0
      self.isFullyExpanded = False


  def set_state(self, state = None):
      """ set the state and variables associated """
      if state == None:
          self.state = self.get_state()
      else:
          self.state = state
      self.tree_number_of_turns = self.state.number_of_turns
      self.isTerminal = self.state.isTerminal()
      if self.isTerminal:
        self.isFullyExpanded = True

  def get_state(self):

      """ recursively obtain the state from root node """
      # reset the state first
      self.reset_state()
      if self.parent == None:
          return self.state
      else:
          state = self.parent.get_state()
          return state.takeAction(self.action)

  def reset_state(self):
      """ reset the state back to that of root """
      if self.parent == None:
          self.state.load_history(self.tree_number_of_turns)
      else:
          self.parent.reset_state()

# A time efficient tree node class for MCTS, which saves the state of game
# at every node using deepcopy, taking up a lot of space, but it is fast, 
# as the game need not be replayed when at a node
class treeNode():
  def __init__(self, state, parent, action):
      self.state = state
      self.tree_number_of_turns = self.state.number_of_turns
      self.isTerminal = state.isTerminal()
      self.isFullyExpanded = self.isTerminal
      self.parent = parent
      self.numVisits = 0
      self.totalReward = 0
      self.children = {}
      self.num_blue_wins = 0
      self.num_red_wins = 0
      self.action = action

# The mcts class that handles all operations of the monte carlo tree search
class mcts():
  def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                rolloutPolicy=randomPolicy, verbose = 1, efficiency = "normal"):
      if timeLimit != None:
          if iterationLimit != None:
              raise ValueError("Cannot have both a time limit and an iteration limit")
          # time taken for each MCTS search in seconds
          self.timeLimit = timeLimit
          self.limitType = 'time'
      else:
          if iterationLimit == None:
              raise ValueError("Must have either a time limit or an iteration limit")
          # number of iterations of the search
          if iterationLimit < 1:
              raise ValueError("Iteration limit must be greater than one")
          self.searchLimit = iterationLimit
          self.limitType = 'iterations'
      self.explorationConstant = explorationConstant
      self.rollout = rolloutPolicy
      self.verbose = verbose
      self.efficiency = efficiency

  def search(self, initialState):

      if self.efficiency == "space":
          self.root = treeNode_spaceEfficient(None, None)
          self.root.set_state(initialState)
      else:
          self.root = treeNode(initialState, None, None)

      if self.limitType == 'time':
          iter = 0
          timeLimit = time.time() + self.timeLimit
          while time.time() < timeLimit:
              self.executeRound()
              iter += 1
          if self.verbose:
            print("{} Seconds: {} iterations ran".format(self.timeLimit, iter))
      else:
          if self.verbose:
              for i in tqdm(range(self.searchLimit), position = 0, leave = True):
                  self.i = i+1
                  self.executeRound()
          else:
              for i in range(self.searchLimit):
                  self.executeRound() 

      bestChild = self.getBestChild(self.root, 0, final_selection = True)
      # self.show_final_results()

      return self.getAction(self.root, bestChild)

  def executeRound(self):

    node = self.selectNode(self.root)
    # set the state for the space efficient method
    if self.efficiency == "space":
        node.set_state()
    reward = self.rollout(node.state)
    self.backpropogate(node, reward)

  def selectNode(self, node):
      """ If node is not fully expanded, it expands it, otherwise it selects best child and then checks for expansion again and etc."""
      while not node.isTerminal:
          if node.isFullyExpanded:
              node = self.getBestChild(node, self.explorationConstant)
          else:
              return self.expand(node)          
      return node

  def expand(self, node):
      # need to set state to get the correct possible actions

      actions = node.state.getPossibleActions()

      for action in actions:
          if action not in node.children:
              if self.efficiency == "space":
                 newNode = treeNode_spaceEfficient(action, node)
              else:
                 newNode = treeNode(copy.deepcopy(node.state.takeAction(action)), node, action)

              node.children[action] = newNode
              if len(actions) == len(node.children):
                  node.isFullyExpanded = True
              return newNode

      raise Exception("Should never reach here")

  def backpropogate(self, node, reward):
      while node is not None:

          node.numVisits += 1
          node.totalReward += reward

          if reward == -1:
            node.num_blue_wins += 1 # blue win
          elif reward == 1:
            node.num_red_wins += 1 # red win

          node.state.load_history(node.tree_number_of_turns)

          node = node.parent

  def show_final_results(self):

      """ Prints the output of the MCTS search """

      for action, node in self.root.children.items():
          print("{}: B:{}, R:{} / {}".format(action,node.num_blue_wins,node.num_red_wins,node.numVisits))


  def getBestChild(self, node, explorationValue, final_selection = False):

      # this check is needed as in the MCTS algo, when choosing the best child in the end, it based on number of visits, not the selection heuristic
      if final_selection == False:
          bestValue = float("-inf")
          bestNodes = []
          for child in node.children.values():
              nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                  2 * math.log(node.numVisits) / child.numVisits)
              if nodeValue > bestValue:
                  bestValue = nodeValue
                  bestNodes = [child]
              elif nodeValue == bestValue:
                  bestNodes.append(child)
          selected_node = random.choice(bestNodes)
          if self.efficiency == "space":
            selected_node.set_state()
          return selected_node
      else:
          # Check for final selection of child 
          bestValue = 0
          bestNodes = []
          for child in node.children.values():
              nodeValue = child.numVisits
              if nodeValue > bestValue:
                  bestValue = nodeValue
                  bestNodes = [child]
              elif nodeValue == bestValue:
                  bestNodes.append(child)
          return random.choice(bestNodes)

  def getAction(self, root, bestChild):
      for action, node in root.children.items():
          if node is bestChild:
              return action
              # get a list of all actions, valid or not
              action_list = self.root.state.getPossibleActions(valid = False)
              # move index recorded for Deep RL training
              self.move_index = action_list.index(action)