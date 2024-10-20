"""
BayesianPSL
"""
 
import torch
import math
from tqdm import tqdm
from torch import Tensor
from typing import Tuple 
from botorch.utils.multi_objective.hypervolume import Hypervolume

import os
import os.path
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_models import GPModelList,GaussianProcess

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
class BayesianPSL(object):
    def __init__(self, problem, MAX_FE: int, BATCH_SIZE: int, x_init: Tensor, y_init: Tensor):
        # problem
        self.problem = problem
        self._lower_x = problem._lower_x
        self._upper_x = problem._upper_x
        self.n_dim = problem.n_dim
        self.n_obj = problem.n_obj
        self.MAX_FE = MAX_FE
        self.BATCH_SIZE = BATCH_SIZE
        self.max_iter = math.ceil((MAX_FE - x_init.shape[0])/BATCH_SIZE)
        self.archive_x = None
        self.archive_y = None
        
        # metric
        self.HV = Hypervolume(ref_point=-problem.ref_point) # minimization problem
        self.hv_list = np.zeros(shape=(0,1),dtype=float)
        
        # initial samples
        if x_init.shape[0] != y_init.shape[0] or x_init.shape[1] != self.n_dim or y_init.shape[1] != self.n_obj:
            raise ValueError(
                "shape of x_init or y_init is incorrect."
            )
        if x_init.shape[0] < self.n_dim:
            raise ValueError(
                "number of intial samples < n_dim."
            )
        if MAX_FE <= x_init.shape[0] or MAX_FE < BATCH_SIZE:
            raise ValueError(
                "setings of MAX_FE and BATCH_SIZE are incorrect."
            )
        self._update_archive(x_init, y_init)
  
    def solve(self):
        print('Iteration: %d, FE: %d HV: %.4f' % (0, self.archive_x.shape[0],self.hv_list[-1,0]))
        for i in tqdm(range(self.max_iter)):
            # one iteration
            new_x, new_obj = self._step()
            print('Iteration: %d, FE: %d HV: %.4f' % (i, self.archive_x.shape[0],self.hv_list[-1,0]))
            
        res = {}
        res['x'] = self.archive_x.detach().numpy()
        res['y'] = self.archive_y.detach().numpy()
        res['FrontNo'] = self.FrontNo
        res['hv'] = self.hv_list
        return res
    
    def _train_psl(self):
        pass
    
    def _batch_selection(self, batch_size: int)->Tensor:
        pass
    
    def _step(self)->Tuple[Tensor,Tensor]:
        # solution normalization x: [0,1]^d, y: [0,1]^m
        self.train_x = torch.div(self.archive_x - self._lower_x,self._upper_x-self._lower_x)
        self.ymin, _ = torch.min(self.archive_y, dim=0)
        self.ymax, _ = torch.max(self.archive_y, dim=0)
        self.train_y = torch.div(torch.sub(self.archive_y, self.ymin), torch.sub(self.ymax, self.ymin))   
        self.z =  -0.1*torch.ones((1,self.n_obj))  
        self.train_y_nds = self.train_y[self.FrontNo[0]].clone()
        
        # train GP surrogate models  
        self.GPModelList =  GPModelList(self.n_obj,self.n_dim, **tkwargs)
        self.GPModelList.fit(self.train_x, self.train_y) 

        # train the psl model
        self._train_psl()   
        
        # greedy batch selection
        batch_size = min(self.MAX_FE - self.archive_x.size(0),self.BATCH_SIZE)
        candidate_x = self._batch_selection(batch_size)
        
        # observe new values
        new_x = (self._upper_x-self._lower_x)*candidate_x + self._lower_x
        new_obj = self.problem.evaluate(new_x)
        self._update_archive(new_x, new_obj)
        return new_x, new_obj
  
    def _update_archive(self, new_x: Tensor, new_obj: Tensor):
        # after add new observations
        if self.archive_x is None:
            self.archive_x = new_x.clone()
            self.archive_y = new_obj.clone()
        else:
            self.archive_x = torch.cat((self.archive_x, new_x),dim=0)
            self.archive_y = torch.cat((self.archive_y, new_obj),dim=0)
        
        # nondominated sorting
        # TODO
        NDSort = NonDominatedSorting()
        self.FrontNo = NDSort.do(self.archive_y.detach().cpu().numpy())
        archive_y_nds = self.archive_y[self.FrontNo[0]].clone()
        # minimization problem
        self.hv_list = np.append(self.hv_list,[[self.HV.compute(-archive_y_nds)]],axis=0)
