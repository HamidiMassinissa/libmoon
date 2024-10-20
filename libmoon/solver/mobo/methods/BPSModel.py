"""
Bayesian Pareto Set Model
"""

import torch
import torch.nn as nn
from torch import Tensor
torch.set_default_dtype(torch.float64)

class BayesianPSModel(torch.nn.Module):
    def __init__(self, n_obj: int, n_dim: int):
        super(BayesianPSModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
         
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
 
       
    def forward(self, pref: Tensor)->Tensor:

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = torch.sigmoid(x) 
        
        return x.to(torch.double)
    
   
    
 