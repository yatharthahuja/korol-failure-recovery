import numpy as np
import torch
import torch.nn as nn
import pdb

class MPCNetwork(nn.Module):
    def __init__(self, robot_state_dim, object_state_dim,
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 seed = 100
                    ):
        super(MPCNetwork, self).__init__()

        self.A_layer_sizes = [robot_state_dim, 64, 128, 64]
        self.B_layer_sizes = [object_state_dim, 32, 64, 64]
        self.merge_layer_sizes = [self.A_layer_sizes[-1]+self.B_layer_sizes[-1], robot_state_dim+object_state_dim]
        # (X_t+1, O_t+1) = f(A * X_t + B * O_t)
        # hidden layers
        torch.manual_seed(seed)
        self.A_layers = nn.ModuleList([nn.Linear(self.A_layer_sizes[i], self.A_layer_sizes[i+1]) \
                         for i in range(len(self.A_layer_sizes) -1)])  # stack severeal layers together.
        self.B_layers = nn.ModuleList([nn.Linear(self.B_layer_sizes[i], self.B_layer_sizes[i+1]) \
                         for i in range(len(self.B_layer_sizes) -1)])  # stack severeal layers together.
        self.merge_layers = nn.ModuleList([nn.Linear(self.merge_layer_sizes[i], self.merge_layer_sizes[i+1]) \
                         for i in range(len(self.merge_layer_sizes) -1)])  # stack severeal layers together.
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh   

    def forward(self, robot_state, object_state):
        robot_out = robot_state
        #pdb.set_trace()
        for i in range(len(self.A_layers)-1):
            robot_out = self.A_layers[i](robot_out)
            robot_out = self.nonlinearity(robot_out)
        robot_out = self.A_layers[-1](robot_out)

        object_out = object_state
        for i in range(len(self.B_layers)-1):
            object_out = self.B_layers[i](object_out)
            object_out = self.nonlinearity(object_out)
        object_out = self.B_layers[-1](object_out)


        out = torch.concat((robot_out, object_out), dim=1)
        for i in range(len(self.merge_layers)-1):
            out = self.merge_layers[i](out)
            out = self.nonlinearity(out)
        out = self.merge_layers[-1](out)
        
        return out