from std_imports import *
from replay_buffer import *
from NN import *

""" 
Agent class 
Purpose 1 : creates and updates neural network 
Purpose 2 : processes output from neural network to decide action for onitama env
Algorithms available: DDPG, D3QN
"""

class Agent:
    
    def __init__(self, model, discount_rate, lr_actor, lr_critic, tau, board_input_shape, card_input_shape, num_actions, 
                 max_mem_size, batch_size, epsilon, epsilon_decay, epsilon_min, update_target, val_constant, 
                 training_name, architecture):
        
        # model
        self.model = model
        
        # discount rate for critic loss (TD error)
        self.discount_rate = discount_rate

        # architecture for neural network
        self.architecture = architecture
        
        # learning rate for actor model
        self.lr_actor = lr_actor
        
        # learning rate for critic model
        self.lr_critic = lr_critic
        
        # input shape for board state 
        self.board_input_shape = board_input_shape
        
        # input shape for card state
        self.card_input_shape = card_input_shape
        
        # stores number of actions
        self.num_actions = num_actions
        
        # constant to scale validity loss
        self.val_constant = val_constant
        
        # DDPG
        if self.model == "DDPG":
            
            # softcopy parameter for target network 
            self.tau = tau
            
            # counter for apply gradients
            self.apply_grad_counter = 0 
            
            # step for apply_grad_counter to hardcopy weights of original to target
            self.update_target = update_target
            
            # memory for replay
            self.memory = replay_buffer(max_mem_size, self.board_input_shape, self.card_input_shape, self.num_actions)
            
            # batch of memory to sample
            self.batch_size = batch_size
            
            # intialise actor model
            self.DDPG_Actor = nn_model(model = "DDPG_Actor", training_name = training_name, learning_rate = self.lr_actor, 
                                       board_input_shape = self.board_input_shape, 
                                       card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                       conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)
            
            # update actor model_names attributes for checkpoints
            self.DDPG_Actor.model_name = "DDPG_Actor"

            # update actor checkpoints_path attributes
            self.DDPG_Actor.checkpoint_path = os.path.join(self.DDPG_Actor.checkpoint_dir, 
                                                           self.DDPG_Actor.model_name + ".pt")
            
             # intialise target actor model
            self.DDPG_Target_Actor = nn_model(model = "DDPG_Actor", training_name = training_name, 
                                              learning_rate = self.lr_actor, board_input_shape = self.board_input_shape, 
                                              card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                              conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)
            
            # update target actor model_names attributes for checkpoints
            self.DDPG_Target_Actor.model_name = "DDPG_Target_Actor"

            # update target actor checkpoints_path attributes
            self.DDPG_Target_Actor.checkpoint_path = os.path.join(self.DDPG_Target_Actor.checkpoint_dir, 
                                                                  self.DDPG_Target_Actor.model_name + ".pt")
            
            # intialise critic model
            self.DDPG_Critic = nn_model(model = "DDPG_Critic", training_name = training_name, 
                                        learning_rate = self.lr_critic, board_input_shape = self.board_input_shape, 
                                        card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                        conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)
            
            # update critic model_names attributes for checkpoints
            self.DDPG_Critic.model_name = "DDPG_Critic"

            # update critic checkpoints_path attributes
            self.DDPG_Critic.checkpoint_path = os.path.join(self.DDPG_Critic.checkpoint_dir, 
                                                            self.DDPG_Critic.model_name + ".pt")
            
            # intialise target critic model
            self.DDPG_Target_Critic = nn_model(model = "DDPG_Critic", training_name = training_name, 
                                               learning_rate = self.lr_critic, board_input_shape = self.board_input_shape, 
                                               card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                               conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)

            # update target critic model_names attributes for checkpoints
            self.DDPG_Target_Critic.model_name = "DDPG_Target_Critic"

            # update target critic checkpoints_path attributes
            self.DDPG_Target_Critic.checkpoint_path = os.path.join(self.DDPG_Target_Critic.checkpoint_dir, 
                                                              self.DDPG_Target_Critic.model_name + ".pt")
            
            # hard update target models' weights to online network to match initialised weights
            self.update_ddpg_target_models(tau = 1)
        
        # D3QN
        elif self.model == "D3QN":
            
            # softcopy parameter for target network 
            self.tau = tau
            
            # list of possible actions (0-39)
            self.actions_list = [x for x in range(self.num_actions)]
            
            # exploration constant
            self.epsilon = epsilon
            
            # decay for exploration constant 
            self.epsilon_decay = epsilon_decay
            
            # minimum exploration constant
            self.epsilon_min = epsilon_min
            
            # batch of memory to sample
            self.batch_size = batch_size
            
            # counter for apply gradients
            self.apply_grad_counter = 0 
            
            # step for apply_grad_counter to hardcopy weights of original to target
            self.update_target = update_target
            
            # memory for replay
            self.memory = replay_buffer(max_mem_size, self.board_input_shape, self.card_input_shape, self.num_actions)
            
            # intialise evaluation model to output q values for actions
            self.D3QN_q_eval = nn_model(model = "D3QN", training_name = training_name, learning_rate = self.lr_actor, 
                                        board_input_shape = self.board_input_shape, 
                                        card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                        conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)
            
            # update q eval attributes for checkpoints
            self.D3QN_q_eval.model_name = "D3QN_q_eval"

            # update critic checkpoints_path attributes
            self.D3QN_q_eval.checkpoint_path = os.path.join(self.D3QN_q_eval.checkpoint_dir, 
                                                            self.D3QN_q_eval.model_name + ".pt")
            
            # intialise target model 
            self.D3QN_q_target = nn_model(model = "D3QN", training_name = training_name, learning_rate = self.lr_actor, 
                                          board_input_shape = self.board_input_shape, 
                                          card_input_shape = self.card_input_shape, num_actions = self.num_actions, 
                                          conv_output_sizes = [64, 128], post_conc_fc_output_sizes = [512, 256], architecture = architecture)
            
            # update q eval attributes for checkpoints
            self.D3QN_q_target.model_name = "D3QN_q_target"

            # update critic checkpoints_path attributes
            self.D3QN_q_target.checkpoint_path = os.path.join(self.D3QN_q_target.checkpoint_dir, 
                                                              self.D3QN_q_target.model_name + ".pt")
            
            # hard update target model's weights to online network to match initialised weights
            self.update_d3qn_target_model(tau = 1)
            
    def select_action(self, mode, board_state, card_state):
        
        """ function to select action for the turn from observations of board and card states """
        
        # DDPG
        if self.model == "DDPG":
        
            # set actor model to evaluation mode (for batch norm and dropout) --> remove instances of batch norm, dropout etc. (things that shd only be around in training)
            self.DDPG_Actor.eval()

            # turn board state to tensor for actor model in device
            board_state = T.tensor(board_state, dtype = T.float).to(self.DDPG_Actor.device)

            # turn card state to tensor for actor model in device
            card_state = T.tensor(card_state, dtype = T.float).to(self.DDPG_Actor.device)

            # feed board state and card state to actor model to obtain softmax probabilities
            softmax_output, _ = self.DDPG_Actor.forward(board_state, card_state, None)

            # obtain numpy array from tensor in device
            action_probs = softmax_output.cpu().detach().numpy()[0]

            # set actor model to training mode (for batch norm and dropout)
            self.DDPG_Actor.train()

            # sample from probabilities during training for exploration
            if mode == 'train':

                action = np.random.choice(self.num_actions, p = action_probs)

            # select argmax during evaluation     
            elif mode == "test":

                action = np.argmax(action_probs)

            return action_probs, action
        
        # D3QN
        elif self.model == "D3QN":
            
            # select action randomly for exploration
            if np.random.random() < self.epsilon and mode != "test":

                action = np.random.choice(self.actions_list)

            # select action greedily for exploitation
            else:
                
                # set D3QN_q_eval model to evaluation mode (for batch norm and dropout) --> remove instances of batch norm, dropout etc. (things that shd only be around in training)
                self.D3QN_q_eval.eval()

                # turn board state to tensor for actor model in device
                board_state = T.tensor(board_state, dtype = T.float).to(self.D3QN_q_eval.device)

                # turn card state to tensor for actor model in device
                card_state = T.tensor(card_state, dtype = T.float).to(self.D3QN_q_eval.device)

                # feed observation tensor to actor model to obtain advantage values
                _ , adv, _ = self.D3QN_q_eval.forward(board_state, card_state, None)

                # if val_actions_mask is present, then the output MUST be multiplied by the validity
                # In other words, all illegal advantages are forced to be zero [HAND OF GOD INTERVENTION]
                if val_actions_mask != None:
                  adv = T.mul(adv, T.tensor(val_actions_mask, dtype = T.float))

                # obtain action with largest advantage value
                action = T.argmax(adv).item()

                # set actor model to training mode (for batch norm and dropout)
                self.D3QN_q_eval.train()

            # obtain one hot encoded action
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[action] = 1

            return action_one_hot, action
            
    def store_memory(self, board_state, card_state, action, val_actions_mask, reward, board_state_prime, card_state_prime, is_done):
    
        """ function to log board_state, card_state, action, reward, board_state_prime, card_state_prime, terminal flag """

        self.memory.log(board_state, card_state, action, val_actions_mask, reward, board_state_prime, card_state_prime, is_done)
        
    def apply_gradients_DDPG(self):
        
        """ function to apply gradients for ddpg """
        """ learns from replay buffer """

        # doesnt not apply gradients if memory does not have at least batch_size number of logs
        if self.memory.mem_counter < self.batch_size:
            return np.nan, np.nan, np.nan, np.nan
        
        # randomly sample batch of memory of board_state, card_state, action, reward, board_state_prime, 
        # card_state_prime, terminal flag from memory log
        board_states, card_states, actions, val_actions_masks, rewards, board_states_prime, card_states_prime, is_dones = \
        self.memory.sample_log(self.batch_size)
        
        # turn features to tensors for critic model in device
        board_states = T.tensor(board_states, dtype = T.float).to(self.DDPG_Critic.device)
        card_states = T.tensor(card_states, dtype = T.float).to(self.DDPG_Critic.device)
        actions = T.tensor(actions, dtype = T.float).to(self.DDPG_Critic.device)
        rewards = T.tensor(rewards, dtype = T.float).to(self.DDPG_Critic.device)
        board_states_prime = T.tensor(board_states_prime, dtype = T.float).to(self.DDPG_Critic.device)
        card_states_prime = T.tensor(card_states_prime, dtype = T.float).to(self.DDPG_Critic.device)
        # turn valid actions targets, binary vector with len = number of actions, with 1 as valid, 0 as illegal, to tensor
        val_actions_masks = T.tensor(val_actions_masks, dtype = T.float).to(self.DDPG_Actor.device)
        
        # set all models to eval mode to calculate td_target
        self.DDPG_Critic.eval()
        self.DDPG_Target_Actor.eval()
        self.DDPG_Target_Critic.eval()

        # obtain actions (softmax) from target actor for board_states_prime and card_states_prime
        target_actions, _ = self.DDPG_Target_Actor.forward(board_states_prime, card_states_prime, None)
        # obtain critic q value by feeding critic with board_states_prime, card_states_prime and target_actions (softmax) 
        target_critic_value = self.DDPG_Target_Critic.forward(board_states_prime, card_states_prime, target_actions)
        
        # obtain critic q value by feeding critic with board_states, card_states and actions (softmax) 
        critic_value = self.DDPG_Critic.forward(board_states, card_states, actions)
        # initialise empty list for td_target
        td_target = []
        # iterate over each batch 
        for i in range(self.batch_size):
            
            # calculate td_target
            td_target.append(rewards[i] + self.discount_rate * target_critic_value[i] * (1 - is_dones[i]))

        # turn td_target to tensor for critic model in device 
        td_target = T.tensor(td_target).to(self.DDPG_Critic.device)
        
        # reshape td_target tensor with batch size as 0th dimension
        td_target = td_target.view(self.batch_size, 1)
        
        # set critic model to train mode 
        self.DDPG_Critic.train()
        
        # reset gradients for critic model to zero
        self.DDPG_Critic.optimizer.zero_grad()
        
        # critic loss is mean squared error between td_target and critic value 
        critic_loss = F.mse_loss(td_target, critic_value)
        
        # critic model back propagation
        critic_loss.backward()
        
        # apply gradients to critic model
        self.DDPG_Critic.optimizer.step()
        
        # set critic to eval mode to calculate actor loss
        self.DDPG_Critic.eval()
        
        # reset gradients for critic model to zero
        self.DDPG_Actor.optimizer.zero_grad()
        
        # obtain actions (softmax) from state following different policy 
        if "actions_only" in self.architecture.lower():
            softmax_output, _ = self.DDPG_Actor.forward(board_states, card_states, None)
            val_output = T.clone(val_actions_masks)
        else:
            softmax_output, val_output = self.DDPG_Actor.forward(board_states, card_states, None)

        # set actor model to train mode 
        self.DDPG_Actor.train()
        
        # gradient ascent using critic value ouput as actor loss
        # loss is coupled with actor model from new_pol_actions 
        actor_training_loss = -self.DDPG_Critic.forward(board_states, card_states, softmax_output)
        
        # reduce mean across batch_size
        actor_training_loss = T.mean(actor_training_loss)
        
        # val loss for actions to actor loss
        actor_val_loss = self.val_constant * F.mse_loss(val_actions_masks, val_output)
        
        # actor loss
        actor_loss = actor_training_loss + actor_val_loss

        # actor model back propagation
        actor_loss.backward()

        # apply gradients to actor model
        self.DDPG_Actor.optimizer.step()
        
        # increment of apply_grad_counter
        self.apply_grad_counter += 1 

        # SOFT COPY OPTION: update target models based on user specified tau
        if self.update_target == None:

             self.update_ddpg_target_models()    

        # HARD COPY OPTION EVERY update_target steps
        else:
            if self.apply_grad_counter % self.update_target == 0: 
            
                self.update_ddpg_target_models(tau = 1)

        return actor_training_loss.item(), actor_val_loss.item(), actor_loss.item(), critic_loss.item()

    def apply_gradients_D3QN(self):

        """ function to apply gradients for d3qn """
        """ learns from replay buffer """
        
        # doesnt not apply gradients if memory does not have at least batch_size number of logs
        if self.memory.mem_counter < self.batch_size:
            return np.nan, np.nan, np.nan
        
        # randomly sample batch of memory of board_state, card_state, action, reward, board_state_prime, 
        # card_state_prime, terminal flag from memory log
        board_states, card_states, actions, val_actions_mask, rewards, board_states_prime, card_states_prime, is_dones = \
        self.memory.sample_log(self.batch_size)
        
        # reset gradients for eval model to zero
        self.D3QN_q_eval.optimizer.zero_grad()
        
        # turn features to tensors for eval model in device
        board_states = T.tensor(board_states, dtype = T.float).to(self.D3QN_q_eval.device)
        card_states = T.tensor(card_states, dtype = T.float).to(self.D3QN_q_eval.device)
        actions = T.tensor(actions, dtype = T.long).to(self.D3QN_q_eval.device)
        rewards = T.tensor(rewards, dtype = T.float).to(self.D3QN_q_eval.device)
        board_states_prime = T.tensor(board_states_prime, dtype = T.float).to(self.D3QN_q_eval.device)
        card_states_prime = T.tensor(card_states_prime, dtype = T.float).to(self.D3QN_q_eval.device)
        is_dones = T.tensor(is_dones, dtype = T.long).to(self.D3QN_q_eval.device)
        
        # turn valid actions targets, binary vector with len = number of actions, with 1 as valid, 0 as illegal, to tensor
        val_actions_mask = T.tensor(val_actions_mask, dtype = T.float).to(self.D3QN_q_eval.device)
        
        # compute v and a from current state using eval model
        v_eval, a_eval, val_eval = self.D3QN_q_eval.forward(board_states, card_states, None)
        
        # compute v and a of next state using eval model
        v_eval_prime, a_eval_prime, _ = self.D3QN_q_eval.forward(board_states_prime, card_states_prime, None)
        
        # compute v and a of next state using target model
        v_target_prime, a_target_prime, _ = self.D3QN_q_target.forward(board_states_prime, card_states_prime, None)
        
        # obtain indices of the batch size
        indices = np.arange(self.batch_size)
        
        # select index of actions from one hot encoding
        actions_index = T.argmax(actions, dim = 1)

        # compute predicted q values for selected actions in current state from eval model
        q_eval = T.add(v_eval, (a_eval - a_eval.mean(dim = 1, keepdim = True))).gather(1, actions_index.unsqueeze(-1)).squeeze(-1)  

        # compute q values for all actions for next state from eval model
        q_eval_prime = T.add(v_eval_prime, (a_eval_prime - a_eval_prime.mean(dim = 1, keepdim = True)))
        
        # compute q values for all actions for next state from target model
        q_target_prime = T.add(v_target_prime, (a_target_prime - a_target_prime.mean(dim = 1, keepdim = True)))
        
        # select maximal actions from q values for next state from eval model
        max_actions = T.argmax(q_eval_prime, dim = 1)

        # print(max_actions)
        #print(q_eval.size())
        
        # mask away q values from terminal 
        q_target_prime[is_dones] = .0
        
        # calculate td target for q
        td_target = rewards + self.discount_rate * q_target_prime.gather(1, max_actions.unsqueeze(-1)).squeeze(-1)          
        
        # print(td_target.size())

        # calculate loss
        training_loss = F.mse_loss(td_target, q_eval) 
        val_loss = self.val_constant * F.mse_loss(val_actions_mask, val_eval)
        loss = training_loss + val_loss

        # actor model back propagation
        loss.backward()
        
        # apply gradients to actor model
        self.D3QN_q_eval.optimizer.step()
        
        # if exploration constant greater than minimum
        if self.epsilon > self.epsilon_min:
            
            # decay
            self.epsilon = self.epsilon - self.epsilon_decay
        
        # else remain as epsilon_min
        else:
            
            self.epsilon = self.epsilon_min
        
        # increment of apply_grad_counter
        self.apply_grad_counter += 1 

        # SOFT COPY OPTION: update target models based on user specified tau
        if self.update_target == None:

             self.update_d3qn_target_model()    

        # HARD COPY OPTION EVERY update_target steps
        else:
            
            if self.apply_grad_counter % self.update_target == 0: 

                self.update_d3qn_target_model(tau = 1)


        return training_loss.item(), val_loss.item(), loss.item()
        
    def update_ddpg_target_models(self, tau = None): 
        
        """ function to soft update target model weights for DDPG. Hard update is possible if tau = 1 """
        
        # use tau attribute if tau not specified 
        if tau is None:
            
            tau = self.tau
        
        # iterate over coupled target actor and actor parameters 
        for target_actor_parameters, actor_parameters in zip(self.DDPG_Target_Actor.parameters(), 
                                                             self.DDPG_Actor.parameters()):
            
            # apply soft update to target actor
            target_actor_parameters.data.copy_((1 - tau) * target_actor_parameters.data + tau * actor_parameters.data)
        
        # iterate over coupled target critic and critic parameters
        for target_critic_parameters, critic_parameters in zip(self.DDPG_Target_Critic.parameters(), 
                                                               self.DDPG_Critic.parameters()):
            # apply soft update to target critic
            target_critic_parameters.data.copy_((1 - tau) * target_critic_parameters.data + tau * critic_parameters.data)
    
    def update_d3qn_target_model(self, tau = None): 
        
        """ function to soft update target model weights for D3QN. Hard update is possible if tau = 1 """
        
        # use tau attribute if tau not specified 
        if tau is None:
            
            tau = self.tau
        
        # iterate over coupled target actor and actor parameters 
        for q_target_parameters, q_eval_parameters in zip(self.D3QN_q_target.parameters(), 
                                                             self.D3QN_q_eval.parameters()):
            
            # apply soft update to target actor
            q_target_parameters.data.copy_((1 - tau) * q_target_parameters.data + tau * q_eval_parameters.data)
    
    def save_all_models(self):
        
        """ save weights for all models """
        
        print("saving model!")
        
        if self.model == "DDPG":
            
            # save weights for each actor, target_actor, critic, target_critic model
            T.save(self.DDPG_Actor.state_dict(), self.DDPG_Actor.checkpoint_path)
            T.save(self.DDPG_Target_Actor.state_dict(), self.DDPG_Target_Actor.checkpoint_path)
            T.save(self.DDPG_Critic.state_dict(), self.DDPG_Critic.checkpoint_path)
            T.save(self.DDPG_Target_Critic.state_dict(), self.DDPG_Target_Critic.checkpoint_path)
        
        elif self.model == "D3QN":
            
            # save weights for q eval and q target model
            T.save(self.D3QN_q_eval.state_dict(), self.D3QN_q_eval.checkpoint_path)
            T.save(self.D3QN_q_target.state_dict(), self.D3QN_q_target.checkpoint_path)

        
    def load_all_models(self):
        
        """ load weights for all models """
        
        print("loading model!")
        
        if self.model == "DDPG":
            
            # load weights for each actor, target_actor, critic, target_critic model
            self.DDPG_Actor.load_state_dict(T.load(self.DDPG_Actor.checkpoint_path))
            self.DDPG_Target_Actor.load_state_dict(T.load(self.DDPG_Target_Actor.checkpoint_path))
            self.DDPG_Critic.load_state_dict(T.load(self.DDPG_Critic.checkpoint_path))
            self.DDPG_Target_Critic.load_state_dict(T.load(self.DDPG_Target_Critic.checkpoint_path))
        
        elif self.model == "D3QN":
            
            # load weights for q eval and q target model
            self.D3QN_q_eval.load_state_dict(T.load(self.D3QN_q_eval.checkpoint_path))
            self.D3QN_q_target.load_state_dict(T.load(self.D3QN_q_target.checkpoint_path))