from std_imports import *
from replay_buffer import *

""" 
Classes and functions to build scalable model
"""
class Multiply(nn.Module):
  def __init__(self):
    super(Multiply, self).__init__()

  def forward(self, tensors):

    result = T.ones(tensors[0].size(), device = T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
    for t in tensors:
      result = result * t

    t = t / T.sum(t)
    return t

class conv_2d_auto_padding(nn.Conv2d):
    
    """ class to set padding dynamically based on kernel size to preserve dimensions of height and width after conv """
    
    def __init__(self, *args, **kwargs):
        
        """ class constructor for conv_2d_auto_padding to alter padding attributes of nn.Conv2d """
        
        # inherit class constructor attributes from nn.Conv2d
        super().__init__(*args, **kwargs)
        
        # dynamically adds padding based on the kernel_size
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

def activation_function(activation):
    
    """ function that returns ModuleDict of activation functions """
    
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['softmax', nn.Softmax(1)],
        ['tanh', nn.Tanh()],
        ['none', nn.Identity()]
    ])[activation]

class vgg_block(nn.Module):
    
    """ class to build basic, vgg inspired block """
    
    def __init__(self, input_channels, output_channels, activation_func, conv, dropout_p, max_pool_kernel):
        
        """ class constructor that creates the layers attributes for vgg_block """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output channels for conv (num of filters)
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # activation function for after batch norm
        self.activation_func = activation_func
        
        # class of conv
        self.conv = conv
        
        # dropout probablity
        self.dropout_p = dropout_p
        
        # size of kernel for maxpooling
        self.max_pool_kernel = max_pool_kernel
        
        # basic vgg_block. input --> conv --> batch norm --> activation func --> dropout --> max pool
        self.block = nn.Sequential(
            
            # conv
            self.conv(self.input_channels, self.output_channels),
            
            # batch norm
            nn.BatchNorm2d(self.output_channels),
            
            # activation func
            activation_function(self.activation_func),
            
            # dropout
            nn.Dropout2d(self.dropout_p),
            
            # maxpooling
            # ceil mode to True to ensure take care of odd dimensions accounted for
            nn.MaxPool2d(self.max_pool_kernel, ceil_mode=True)
            
        )
    
    def forward(self, x):
        
        """ function for forward pass of vgg_block """
        
        x = self.block(x)
        
        return x

class fc_block(nn.Module):
    
    """ class to build basic fc block """
    
    def __init__(self, input_shape, output_shape, activation_func, dropout_p):
        
        """ class constructor that creates the layers attributes for fc_block """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output units for hidden layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # activation function for after batch norm
        self.activation_func = activation_func 
        
        # dropout probablity
        self.dropout_p = dropout_p
        
        # basic fc_block. inpuit --> linear --> batch norm --> activation function --> dropout 
        self.block = nn.Sequential(
            
            # linear hidden layer
            nn.Linear(self.input_shape, self.output_shape, bias = False),
            
            # batch norm
            nn.BatchNorm1d(self.output_shape),
            
            # activation func
            activation_function(self.activation_func),
            
            # dropout
            nn.Dropout(self.dropout_p),
            
        )
    
    def forward(self, x):
        
        """ function for forward pass of fc_block """
        
        x = self.block(x)
        
        return x

class nn_layers(nn.Module):
    
    """ class to build layers of vgg_block or fc_block """
    
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input channels/shape
        self.input_channels = input_channels
        
        # class of block
        self.block = block
        
        # output channels/shape
        self.output_channels = output_channels
        self.input_output_list = list(zip(output_channels[:], output_channels[1:]))
        
        # module list of layers with same args and kwargs
        self.blocks = nn.ModuleList([
            
            self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
            *[self.block(input_channels, output_channels, *args, **kwargs) 
            for (input_channels, output_channels) in self.input_output_list]   
            
        ])
    
    def get_flat_output_shape(self, input_shape):
        
        """ function to obatain number of features after flattening after convolution layers """
        
        # initialise dummy tensor of ones with input shape
        x = T.ones(1, *input_shape)
        
        # feed dummy tensor to blocks by iterating over each block
        for block in self.blocks:
            
            x = block(x)
        
        # flatten resulting tensor and obtain features after flattening
        n_size = x.view(1, -1).size(1)
        
        return n_size
    
    def forward(self, x):
        
        """ function for forward pass of layers """
        
        # iterate over each block
        for block in self.blocks:
            
            x = block(x)
            
        return x 
    
class nn_model(nn.Module):
    
    """ class to build model for DDPG, D3QN """
    
    def __init__(self, model, training_name, learning_rate, board_input_shape, card_input_shape, num_actions, 
                 conv_output_sizes, post_conc_fc_output_sizes, architecture):
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = None
        
        # checkpoint filepath 
        self.checkpoint_path = None

        # create directory for saving models
        # os.mkdir("Saved_Models/" + training_name + "_" + "best_models/")
        try:
          os.mkdir("Saved_Models/" + training_name + "_" + "best_models/")
        except:
            shutil.rmtree("Saved_Models/" + training_name + "_" + "best_models/")
            os.mkdir("Saved_Models/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "Saved_Models/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
        
        # input shape for board state 
        self.board_input_shape = board_input_shape
        
        # input shape for card state
        self.card_input_shape = card_input_shape
        
        # input shape for number of actions
        self.num_actions = num_actions
        
        # list of sizes of output channels for conv in vgg layer
        self.conv_output_sizes = conv_output_sizes
        
        # list of sizes of output shape for fc in fc layer post concatenation
        self.post_conc_fc_output_sizes = post_conc_fc_output_sizes

        # specified architecture to handle validity
        self.architecture = architecture

        if "dual" in architecture.lower():

            # conv layers pre concatenation for board state
            self.board_conv_layers_actions = nn_layers(input_channels = board_input_shape[0], block = vgg_block, output_channels = 
                                               self.conv_output_sizes, activation_func = 'relu', conv = 
                                               partial(conv_2d_auto_padding, kernel_size = 3, bias = False) , dropout_p = 0, 
                                               max_pool_kernel = 2)

            # single fc block pre concatenation for card state
            self.card_fc_block_actions = fc_block(input_shape = card_input_shape, output_shape = card_input_shape, 
                                               activation_func = 'relu', dropout_p = 0)

            # conv layers pre concatenation for board state
            self.board_conv_layers_val = nn_layers(input_channels = board_input_shape[0], block = vgg_block, output_channels = 
                                               self.conv_output_sizes, activation_func = 'relu', conv = 
                                               partial(conv_2d_auto_padding, kernel_size = 3, bias = False) , dropout_p = 0, 
                                               max_pool_kernel = 2)

            # single fc block pre concatenation for card state
            self.card_fc_block_val = fc_block(input_shape = card_input_shape, output_shape = card_input_shape, 
                                               activation_func = 'relu', dropout_p = 0)

        else:
            # conv layers pre concatenation for board state
            self.board_conv_layers = nn_layers(input_channels = board_input_shape[0], block = vgg_block, output_channels = 
                                               self.conv_output_sizes, activation_func = 'relu', conv = 
                                               partial(conv_2d_auto_padding, kernel_size = 3, bias = False) , dropout_p = 0, 
                                               max_pool_kernel = 2)

            # single fc block pre concatenation for card state
            self.card_fc_block = fc_block(input_shape = card_input_shape, output_shape = card_input_shape, 
                                               activation_func = 'relu', dropout_p = 0)
        
        if self.model == "DDPG_Critic":
            
            if "dual" in architecture.lower():

                self.post_conc_fc_layers = nn_layers(input_channels = self.card_input_shape + self.num_actions + \
                                                     self.board_conv_layers_actions.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)
            else:
                # fc layers post concatenation
                self.post_conc_fc_layers = nn_layers(input_channels = self.card_input_shape + self.num_actions + \
                                                     self.board_conv_layers.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)
            
            # single fc block pre concatenation for actions
            self.actions_fc_block = fc_block(input_shape = num_actions, output_shape = num_actions, 
                                             activation_func = 'relu', dropout_p = 0)

            # final single fc block post concatenation to output state action value (q) without any activation function
            self.q_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = 1, 
                                     activation_func = 'none', dropout_p = 0)
        
        elif self.model == "DDPG_Actor":
            
            if "dual" in architecture.lower():

                self.post_conc_fc_layers_actions = nn_layers(input_channels = self.card_input_shape + \
                                                     self.board_conv_layers_actions.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)

                self.post_conc_fc_layers_val = nn_layers(input_channels = self.card_input_shape + \
                                                     self.board_conv_layers_val.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)

                self.softmax_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions, 
                                           activation_func = 'softmax', dropout_p = 0)

                self.val_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions,
                                             activation_func = 'sigmoid', dropout_p = 0)

                self.multiply = Multiply()

            elif "actions_only" in architecture.lower():

                self.post_conc_fc_layers = nn_layers(input_channels = self.card_input_shape + \
                                                     self.board_conv_layers.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)

                self.softmax_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions, 
                                               activation_func = 'softmax', dropout_p = 0)

            else:
                # fc layers post concatenation
                self.post_conc_fc_layers = nn_layers(input_channels = self.card_input_shape + \
                                                     self.board_conv_layers.get_flat_output_shape(self.board_input_shape), 
                                                     block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                     activation_func = 'relu', dropout_p = 0)
                
                # single fc block post concatenation to output softmaxed actions
                if "val_after_actions" in architecture.lower() or "branch" in architecture.lower():
                    self.softmax_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions, 
                                               activation_func = 'softmax', dropout_p = 0)
                elif "actions_after_val" in architecture.lower():
                    self.softmax_output = fc_block(input_shape = num_actions, output_shape = num_actions, 
                                               activation_func = 'softmax', dropout_p = 0)
                
                # single fc block post concatenation to output validity of actions
                if "actions_after_val" in architecture.lower() or "branch" in architecture.lower():
                    self.val_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions,
                                                 activation_func = 'sigmoid', dropout_p = 0)
                elif "val_after_actions" in architecture.lower():
                    self.val_output = fc_block(input_shape = num_actions, output_shape = num_actions,
                                                 activation_func = 'sigmoid', dropout_p = 0)
                # multiply layer 
                if "multiply" in architecture.lower():
                    self.multiply = Multiply()

        
        elif self.model == "D3QN":
            
            # fc layers post concatenation
            self.post_conc_fc_layers = nn_layers(input_channels = self.card_input_shape + \
                                                 self.board_conv_layers.get_flat_output_shape(self.board_input_shape), 
                                                 block = fc_block, output_channels = self.post_conc_fc_output_sizes, 
                                                 activation_func = 'relu', dropout_p = 0)
            
            # single fc block post concatenation to output advantage value for each action
            self.a_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = num_actions, 
                                           activation_func = 'none', dropout_p = 0)
            
            # single fc block post concatenation to output state value (v) layer
            self.v_output = fc_block(input_shape = self.post_conc_fc_output_sizes[-1], output_shape = 1, 
                                     activation_func = 'none', dropout_p = 0)
            
            # final single fc block post concatenation to output validity of actions
            self.val_output = fc_block(input_shape = num_actions, output_shape = num_actions, activation_func = 'sigmoid', 
                                       dropout_p = 0)
            
        # adam optimizer 
        self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
        
    def forward(self, board_state, card_state, actions):
            
        """ function for forward pass through critic_model """
        
        # forward pass for ddpg critic model
        if self.model == "DDPG_Critic":
            
            if "dual" in self.architecture.lower():
                # board_state --> conv --> flatten 
                board_state = self.board_conv_layers_actions(board_state)
                board_state_flat = board_state.view(board_state.size(0), -1)

                # card_state --> linear 
                card_state = self.card_fc_block_actions(card_state)
            else:
                # board_state --> conv --> flatten 
                board_state = self.board_conv_layers(board_state)
                board_state_flat = board_state.view(board_state.size(0), -1)

                # card_state --> linear 
                card_state = self.card_fc_block(card_state)

            # actions --> linear 
            actions = self.actions_fc_block(actions)

            # concatenate intermediate tensors
            conc = T.cat((board_state_flat, card_state, actions), 1)

            # intermediate tensor --> fc linear layers
            conc = self.post_conc_fc_layers(conc)

            # fc linear layers --> final fc block --> q value
            q = self.q_output(conc)

            return q
        
        # forward pass for ddpg actor model
        elif self.model == "DDPG_Actor":

            if "dual" in self.architecture.lower():
                
                # board_state --> conv --> flatten 
                board_state_actions = self.board_conv_layers_actions(board_state)
                board_state_flat_actions = board_state_actions.view(board_state_actions.size(0), -1)

                # card_state --> linear 
                card_state_actions = self.card_fc_block_actions(card_state)

                # concatenate intermediate tensors
                conc_actions = T.cat((board_state_flat_actions, card_state_actions), 1)

                # board_state --> conv --> flatten 
                board_state_val = self.board_conv_layers_val(board_state)
                board_state_flat_val = board_state_val.view(board_state_val.size(0), -1)

                # card_state --> linear 
                card_state_val = self.card_fc_block_val(card_state)

                # concatenate intermediate tensors
                conc_val = T.cat((board_state_flat_val, card_state_val), 1)

                # intermediate tensor --> actions fc linear layers
                conc_actions = self.post_conc_fc_layers_actions(conc_actions)

                # intermediate tensor --> actions fc linear layers
                conc_val = self.post_conc_fc_layers_val(conc_val)

                actions = self.softmax_output(conc_actions)

                val = self.val_output(conc_val)

                actions = self.multiply([actions, val])

            elif "actions_only" in self.architecture.lower():

                board_state = self.board_conv_layers(board_state)
                board_state_flat = board_state.view(board_state.size(0), -1)

                # card_state --> linear 
                card_state = self.card_fc_block(card_state)

                # concatenate intermediate tensors
                conc = T.cat((board_state_flat, card_state), 1)

                # intermediate tensor --> actions fc linear layers
                conc = self.post_conc_fc_layers(conc)

                actions = self.softmax_output(conc)

                # dummy val variable
                val = T.clone(actions)

            else:

                # board_state --> conv --> flatten 
                board_state = self.board_conv_layers(board_state)
                board_state_flat = board_state.view(board_state.size(0), -1)

                # card_state --> linear 
                card_state = self.card_fc_block(card_state)

                # concatenate intermediate tensors
                conc = T.cat((board_state_flat, card_state), 1)

                # intermediate tensor --> fc linear layers
                conc = self.post_conc_fc_layers(conc)

                # determine the next part of the architecture
                if "val_after_actions" in self.architecture.lower():
                    actions = self.softmax_output(conc)
                    val = self.val_output(actions)
                elif "actions_after_val" in self.architecture.lower():
                    val = self.val_output(conc)
                    actions = self.softmax_output(val)
                elif "branch" in self.architecture.lower():
                    actions = self.softmax_output(conc)
                    val = self.val_output(conc)

                if "multiply" in self.architecture.lower():
                    # multiply validity to actions
                    actions = self.multiply([actions, val])
            
            return actions, val
        
        # forward pass for d3qn
        elif self.model == "D3QN":
            
            # board_state --> conv --> flatten 
            board_state = self.board_conv_layers(board_state)
            board_state_flat = board_state.view(board_state.size(0), -1)

            # card_state --> linear 
            card_state = self.card_fc_block(card_state)
            
            # concatenate intermediate tensors
            conc = T.cat((board_state_flat, card_state), 1)
            
            # intermediate tensor --> fc linear layers
            conc = self.post_conc_fc_layers(conc)
            
            # fc linear layers --> state value of actions
            v = self.v_output(conc)
            
            # fc linear layers --> advantage value of actions
            a = self.a_output(conc)
            
            # advantage value of actions --> validity of actions 
            val = self.val_output(a)
            
            return v, a, val