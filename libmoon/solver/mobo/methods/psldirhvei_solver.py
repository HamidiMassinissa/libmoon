"""
PSL + DirHV-EI

[1] Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang. Pareto Set Learning for
Expensive Multiobjective Optimization. Advances in Neural Information Processing
Systems (NeurIPS) , 2022
[2] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel 
Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
Computation, 28(2): 432-444, 2024.
"""

import torch
from torch import Tensor 
from typing import Tuple 

from botorch.utils.sampling import sample_simplex
from botorch.utils.probability.utils import (
    ndtr as Phi, # Standard normal CDF
    phi, # Standard normal PDF
)

import os
import os.path
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.BPSModel import BayesianPSModel as ParetoSetModel
from methods.base_solver_BPSL import BayesianPSL

class PSLDirHVEISolver(BayesianPSL):
    def __init__(self, problem, MAX_FE: int, BATCH_SIZE: int, x_init: Tensor, y_init: Tensor):
        super().__init__(problem, MAX_FE, BATCH_SIZE, x_init, y_init)
        self.solver_name = 'psldirhvei'
        # Parameter Setting
        self.n_steps = 1000 # number of learning steps
        self.n_pref_update = 10 # number of sampled preferences per step
        self.coef_lcb = 0.5 # coefficient of LCB
        self.n_candidate = 1000  # number of sampled candidates on the approxiamte PF
        self.learning_rate = 1e-3


    def _get_xis(self, ref_vecs: Tensor)->Tuple[Tensor,Tensor]:
        # ref_vecs is generated via simplex-lattice design
        # temp = 1.1 * ref_vecs - self.z
        dir_vecs = ref_vecs / torch.norm(ref_vecs, dim=1, keepdim=True)
        # Eq. 11, compute the intersection points
        div_dir = 1.0 / dir_vecs
        A = self.train_y_nds - self.z  # L*M
        G = torch.ger(div_dir[:, 0], A[:, 0])  # N*L, f1
        for j in range(1, self.n_obj):
            G = torch.max(G, torch.ger(div_dir[:, j], A[:, j]))  # N*L, max(fi,fj)
        
        # minimum of mTch for each direction vector
        Lmin = torch.min(G, dim=1, keepdim=True).values.data  # N*1  one for each direction vector
        # N*M  Intersection points
        xis = self.z + torch.mul(Lmin, dir_vecs)
        return xis, dir_vecs
	
    def _train_psl(self):
        self.psmodel = ParetoSetModel(self.n_obj, self.n_dim)
        self.psmodel.to(**tkwargs)
        # optimizer
        optimizer = torch.optim.Adam(self.psmodel.parameters(), lr=self.learning_rate)
        # t_step Pareto Set Learning with Gaussian Process
        for t_step in range(self.n_steps):
            self.psmodel.train()
            
            # sample n_pref_update preferences L1=1
            pref_vec = sample_simplex(d=self.n_obj, n=self.n_pref_update-self.n_obj).to(**tkwargs)
            pref_vec = torch.cat((pref_vec, torch.eye(self.n_obj).to(**tkwargs)),dim=0)
            pref_vec = torch.clamp(pref_vec, min=1.e-6) 
            
            xis, dir_vecs = self._get_xis(pref_vec) 
            # get the current coressponding solutions
            x = self.psmodel(pref_vec)
            mean, std, mean_grad, std_grad = self.GPModelList.evaluate(x, calc_std=True, calc_gradient=True) 
            
            xi_minus_u = xis - mean  # N*M
            tau = xi_minus_u / std  # N*M
            alpha_i = xi_minus_u * Phi(tau) + std * phi(tau)  # N*M
   
            alpha_mean_grad = (-Phi(tau)*alpha_i).unsqueeze(2).repeat(1,1,self.n_dim) * mean_grad
            alpha_std_grad = (phi(tau)*alpha_i).unsqueeze(2).repeat(1,1,self.n_dim) * std_grad
            dirhvei_grad = -torch.sum( alpha_mean_grad + alpha_std_grad, dim=1)
             
            # gradient-based pareto set model update 
            optimizer.zero_grad()
            self.psmodel(pref_vec).backward(dirhvei_grad)
            optimizer.step()  
            
    def _batch_selection(self, batch_size: int)->Tensor:
        # sample n_candidate preferences default:1000
        self.psmodel.eval()  # Sets the module in evaluation mode.
        pref = sample_simplex(d=self.n_obj, n=self.n_candidate).to(**tkwargs)
        pref = torch.clamp(pref, min=1.e-6) 
        
        # generate correponding solutions, get the predicted mean/std
        with torch.no_grad():
            candidate_x = self.psmodel(pref).to(**tkwargs)
            candidate_mean, candidata_std = self.GPModelList.evaluate(candidate_x, calc_std=True, calc_gradient=False) 
        xis, dir_vecs = self._get_xis(pref)  
        EIDs = torch.zeros(self.n_candidate,self.n_candidate).to(**tkwargs)
        for i in range(self.n_candidate):
            temp_mean = candidate_mean[i:i+1].repeat(self.n_candidate,1)
            temp_std = candidata_std[i:i+1].repeat(self.n_candidate,1)
            xi_minus_u = xis - temp_mean  # N*M
            tau = xi_minus_u / temp_std  # N*M
            alpha_i = xi_minus_u * Phi(tau) + temp_std * phi(tau)  # N*M
            EIDs[i,:] = torch.prod(alpha_i, dim=1)  
        Qb = []
        temp = EIDs.clone()
        beta = torch.zeros(self.n_candidate).to(**tkwargs)  
        for i in range(batch_size):
            index = torch.argmax(torch.sum(temp, dim=1))
            Qb.append(index.item())
            beta = beta + temp[index, :]
            # Update temp: [EI_D(x|\lambda) - beta]_+
            temp = EIDs - beta[None, :].repeat(self.n_candidate, 1)
            temp[temp < 0] = 0
  
        # evaluate the selected n_sample solutions
        new_x = candidate_x[Qb]
        return new_x
    
if __name__ == '__main__':
    import time
    from utils.lhs import lhs
    import matplotlib.pyplot as plt
    from test_functions import ZDT1,ZDT2,ZDT3,ZDT4,ZDT6
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    # minimization
    problem = ZDT2(n_obj=2,n_dim=8)
    n_init = 11*problem.n_dim-1
    batch_size = 5
    maxFE = 200
    ts = time.time()
 
    x_init = torch.from_numpy(lhs(problem.n_dim, samples=n_init)).to(**tkwargs)
    y_init = problem.evaluate(x_init)
    solver = PSLDirHVEISolver(problem, maxFE, batch_size, x_init, y_init)
    solver.debug = True
    res = solver.solve()
    elapsed = time.time() - ts
    res['elapsed'] = elapsed

    fig = plt.figure()
    plt.scatter(res['y'][res['FrontNo'][0],0], res['y'][res['FrontNo'][0],1], label='Solutions')
    if hasattr(problem, '_get_pf'):
        plt.plot(problem._get_pf()[:,0], problem._get_pf()[:,1], label='PF')

    plt.legend(fontsize=16)
    plt.xlabel('$f_1$', fontsize=18)
    plt.ylabel('$f_2$', fontsize=18)
    plt.show()
    
     
    
   
    
 