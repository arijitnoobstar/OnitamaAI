from std_imports import *

""" 
The replay_buffer class
Purpose 1: store memory of state, action, state_prime, reward, terminal flag 
Purpose 2: function to randomly sample a batch of memory
"""

class replay_buffer:
    
    def __init__(self, max_mem_size, board_state_input_shape, card_state_input_shape, action_space):
        
        """ class constructor that initialises memory states attributes """
        
        # bound for memory log
        self.mem_size = max_mem_size
        
        # counter for memory logged
        self.mem_counter = 0 
        
        # logs for board_state, card_state, action, board_state_prime, card_state_prime, reward, terminal flag
        # board state should be array of shape [[(no. of channels/depth), height, width] for CNN 
        # (binary board for master + binary board for pawns) * 2 for 2 players --> [4, 5, 5]
        # card state should be one hot encoded of 16 cards for current and opposing player and side card --> 48 features
        # action space stores softmax of 5 * 2 * 4 = 40 actions
        self.board_state_log = np.zeros((self.mem_size, *board_state_input_shape))
        self.board_state_prime_log = np.zeros((self.mem_size, *board_state_input_shape))
        self.card_state_log = np.zeros((self.mem_size, card_state_input_shape))
        self.card_state_prime_log = np.zeros((self.mem_size, card_state_input_shape))
                    
        self.action_log = np.zeros((self.mem_size, action_space))
        self.val_actions_log = np.zeros((self.mem_size, action_space))
            
        self.reward_log = np.zeros(self.mem_size)
        self.terminal_log = np.zeros(self.mem_size)
        
    def log(self, board_state, card_state, action, val_actions_target, reward, board_state_prime, card_state_prime, is_done):
        
        """ log memory """
        
        # index for logging. based on first in first out
        index = self.mem_counter % self.mem_size
        
        # log memory for state, action, state_prime, reward, terminal flag
        self.board_state_log[index] = board_state
        self.board_state_prime_log[index] = board_state_prime
        self.card_state_log[index] = card_state
        self.card_state_prime_log[index] = card_state_prime
        self.action_log[index] = action
        self.val_actions_log[index] = val_actions_target
        self.reward_log[index] = reward
        self.terminal_log[index] = is_done

        # increment counter
        self.mem_counter += 1
    
    def sample_log(self, batch_size):
        
        """ function to randomly sample a batch of memory """
        
        # select amongst memory logs that is filled
        max_mem = min(self.mem_counter, self.mem_size)
        
        # randomly select memory from logs
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        # obtain corresponding state, action, state_prime, reward, terminal flag
        board_states = self.board_state_log[batch]
        board_states_prime = self.board_state_prime_log[batch]
        card_states = self.card_state_log[batch]
        card_states_prime = self.card_state_prime_log[batch]
        actions = self.action_log[batch]
        val_actions_targets = self.val_actions_log[batch]
        rewards = self.reward_log[batch]
        is_dones = self.terminal_log[batch]
        
        return board_states, card_states, actions, val_actions_targets, rewards, board_states_prime, card_states_prime, is_dones