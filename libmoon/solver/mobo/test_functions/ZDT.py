#! /usr/bin/env python3
"""
ZDT, Multi-objective optimization benchmark problems.
 
Reference [Zitzler2000] 
-- E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
   evolutionary algorithms: Empirical results, Evolutionary computation,
   2000, 8(2): 173-195.
"""

from __future__ import annotations

import math
from typing import List, Union

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
 
class ZDT():
    def __init__(
        self,
        n_obj: int = 2,
        n_dim: int = 8,
    ) -> None:
        if n_dim <= n_obj:
            raise ValueError(
                f"n_dim must > n_obj, but got {n_dim} and {n_obj}."
            )
        self.n_obj = n_obj
        self.n_dim = n_dim
    
        self._lower_x = torch.zeros((1,self.n_dim)).to(**tkwargs) 
        self._upper_x = torch.ones((1,self.n_dim)).to(**tkwargs)
        self.ref_point = 11*torch.ones(self.n_obj).to(**tkwargs)


class ZDT1(ZDT):

    def evaluate(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = 1 + 9 * X[..., 1:].mean(dim=-1)
        f_1 = g * (1 - (f_0 / g).sqrt())
        return torch.stack([f_0, f_1], dim=-1)
    
    def _get_pf(self, n_points: int = 100):
        f1 = torch.linspace(0, 1, n_points)
        f2 = 1 - torch.sqrt(f1)
        return torch.stack((f1, f2), axis=1)

   

class ZDT2(ZDT):

    def evaluate(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = 1 + 9 * X[..., 1:].mean(dim=-1)
        f_1 = g * (1 - (f_0 / g).pow(2))
        return torch.stack([f_0, f_1], dim=-1)
    
    def _get_pf(self, n_points: int = 100):
        f1 = torch.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        return torch.stack((f1, f2), axis=1)

   

class ZDT3(ZDT):

    def evaluate(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = 1 + 9 * X[..., 1:].mean(dim=-1)
        f_1 = g * (1 - (f_0 / g).sqrt() - f_0 / g * torch.sin(10 * math.pi * f_0))
        return torch.stack([f_0, f_1], dim=-1)
    
    def _get_pf(self, n_points: int = 100):
        f1 = torch.hstack([torch.linspace(0, 0.0830, int(n_points / 5)),
                        torch.linspace(0.1822, 0.2578, int(n_points / 5)),
                        torch.linspace(0.4093, 0.4539, int(n_points / 5)),
                        torch.linspace(0.6183, 0.6525, int(n_points / 5)),
                        torch.linspace(0.8233, 0.8518, n_points - 4 * int(n_points / 5))])
        f2 = 1 - torch.sqrt(f1) - f1 * torch.sin(10 * math.pi * f1)
        return torch.stack((f1, f2), axis=1)

    
    
class ZDT6(ZDT):

    def evaluate(self, X: Tensor) -> Tensor:
        f_0 = 1 - torch.exp(-4*X[...,0])*torch.pow(torch.sin(6*math.pi*X[...,0]),6) 
        g = 1 + 9 * torch.pow(X[..., 1:].mean(dim=-1),0.25) 
        f_1 = g * (1 - (f_0 / g).pow(2))
        return torch.stack([f_0, f_1], dim=-1)
    
    def _get_pf(self, n_points: int = 100):
        f1 = torch.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        return torch.stack((f1, f2), axis=1)

  

