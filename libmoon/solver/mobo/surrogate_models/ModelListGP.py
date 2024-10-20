# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import Optional, Tuple 


from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel

from botorch.models.gp_regression import SingleTaskGP  
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_mll
 
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
 

class GPModelList():
    def __init__(self, n_obj: int, n_dim: int, **kwargs):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.tkwargs = kwargs
   
    def fit(self, train_x: Tensor, train_y: Tensor):
        if train_x.dim()!= 2 or train_y.dim()!=2:
            raise ValueError(
                "dim of train_x and train_y should be 2" 
            )
        if train_x.shape[1] != self.n_dim or train_y.shape[1] != self.n_obj or train_x.shape[0] != train_y.shape[0]:
            raise ValueError(
                "train_x should have shape [:, {}] and train_y should have shape [:, {}]".format(self.n_dim, self.n_obj)
            )
        # define models for objective and constraint
        train_x, train_y = train_x.to(**self.tkwargs), train_y.to(**self.tkwargs)
        models = []
        for i in range(self.n_obj):
            train_obj = train_y[..., i:i+1]
            train_yvar = torch.full_like(train_obj, 1e-4) # noise-free setting
            models.append(
                SingleTaskGP(train_x, train_obj, train_yvar,
                             covar_module = ScaleKernel(RBFKernel()),
                             outcome_transform=Standardize(m=1))
            )
        self.model = ModelListGP(*models)
        self.model.to(**self.tkwargs)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll) 
             
     
     
    
    def evaluate(self, test_x: Tensor, calc_std: bool = True, calc_gradient: bool=False
                       ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # test_x: N * n_var
        # mean: N*n_obj, std: N*n_obj
        # mean_grad: N*n_obj*n_var, std_grad: N*n_obj*n_var
        if test_x.dim()!= 2:
            raise ValueError(
                "dim of test_x should be 2" 
            )
        if test_x.shape[1] != self.n_dim:
            raise ValueError(
                "test_x should have shape [:, {}] and not [:, {}]".format(self.n_dim, test_x.shape[1])
            )
        min_var = 0
        posterior = self.model.posterior(X=test_x)
        mean = posterior.mean # N*n_obj
        
        if not calc_std:
            return mean 
        
        std = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape) # N*n_obj
        if not calc_gradient:
            return mean, std 
         
        mean_grad = torch.zeros(test_x.size(0), self.n_obj, self.n_dim, **self.tkwargs)
        std_grad = torch.zeros(test_x.size(0), self.n_obj, self.n_dim, **self.tkwargs)
             
        for j in range(self.n_obj):
            mean_grad[:,j,:] = torch.autograd.grad(mean[:,j].sum(), test_x, retain_graph=True)[0]
            std_grad[:,j,:] = torch.autograd.grad(std[:,j].sum(), test_x, retain_graph=True)[0] 
 
        return mean, std, mean_grad, std_grad