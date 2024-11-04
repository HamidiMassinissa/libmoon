'''
    Main algorithm framework for  Decomposition-based Multi-objective Bayesian Optimization.

    [1] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel
    Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary
    Computation, 28(2): 432-444, 2024.
'''

import torch
import math
from tqdm import tqdm
from torch import Tensor
from typing import Tuple 
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.probability.utils import (
    ndtr as Phi, # Standard normal CDF
    phi, # Standard normal PDF
)
import matplotlib.pyplot as plt
import os
import os.path
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_models import GPModelList
from utils.lhs import lhs

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class DirHVEGOSolver(object):
    def __init__(self, problem, MAX_FE: int, BATCH_SIZE: int, x_init: Tensor, y_init: Tensor):
        self.solver_name = 'DirHV-EGO'
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
        self.debug = False
        
        # metric
        self.HV = Hypervolume(ref_point=-problem.ref_point) # minimization problem
        self.hv_list = np.zeros(shape=(0,1),dtype=float)
        self.FE = np.zeros(shape=(0,1),dtype=float)
        
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
        
        # generate reference vectors
        params_H =  [199,19] # parameter for weight vectors, for M = 2,3 respectively.
        if self.n_obj == 2 or self.n_obj == 3:
            # simplex-lattice design
            weights = get_reference_directions("uniform", self.n_obj, n_partitions=params_H[self.n_obj-2])
            self.ref_vecs = torch.from_numpy(weights).to(**tkwargs)
        else:
            # TODO
            pass
        
    def solve(self):
        print('Iteration: %d, FE: %d HV: %.4f' % (0, self.archive_x.shape[0],self.hv_list[-1,0]))
        for i in tqdm(range(self.max_iter)):
            # one iteration
            new_x, new_obj = self._step()
            print('Iteration: %d, FE: %d HV: %.4f' % (i, self.archive_x.shape[0],self.hv_list[-1,0]))
            
        res = {}
        res['x'] = self.archive_x.to('cpu')
        res['y'] = self.archive_y.to('cpu')
        res['FrontNo'] = self.FrontNo
        res['hv'] = self.hv_list
        return res
        
    def _get_acquisition(self, u: Tensor, sigma: Tensor, ref_vec: Tensor, pref_inc: Tensor):
        '''
        Parameters:
            ref_vec: direction vector
            pref_inc :  preference-conditional incumbent

        Returns
            preference-conditional EI: DirHV-EI(X|pref_vec)

        '''
        xi_minus_u = pref_inc - u  # N*M
        tau = xi_minus_u / sigma  # N*M
        alpha_i = xi_minus_u * Phi(tau) + sigma * phi(tau)  # N*M
        return torch.prod(alpha_i, dim=1)
    
    def _get_xis(self, ref_vecs: Tensor)->Tuple[Tensor,Tensor]:
        # ref_vecs is generated via simplex-lattice design
        temp = 1.1 * ref_vecs - self.z
        dir_vecs = temp / torch.norm(temp, dim=1, keepdim=True)
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
     
       
    def _step(self)->Tuple[Tensor,Tensor]:
        # solution normalization x: [0,1]^d, y: [0,1]^m
        self.train_x = self.archive_x.clone()
        self.ymin, _ = torch.min(self.archive_y, dim=0)
        self.ymax, _ = torch.max(self.archive_y, dim=0)
        self.train_y = torch.div(torch.sub(self.archive_y, self.ymin), torch.sub(self.ymax, self.ymin))   
        self.z =  -0.01*torch.ones((1,self.n_obj)).to(**tkwargs)  
        self.train_y_nds = self.train_y[self.FrontNo[0]].clone()
        
        # train GP surrogate models  
        self.GaussianProcess =  GPModelList(self.n_obj,self.n_dim, **tkwargs)
        self.GaussianProcess.fit(self.train_x, self.train_y) 
        
        # maximizing the preference conditional acquisition functions
        # Calculate the Intersection points and Direction vectors
        xis, dir_vecs = self._get_xis(self.ref_vecs)
        # Use MOEA/D to maximize DirHV-EI
        candidate_x, candidate_mean, candidata_std = self._moead_gr(dir_vecs, xis)
        
        
        # greedy batch selection
        batch_size = min(self.MAX_FE - self.archive_x.size(0),self.BATCH_SIZE)
        
        # Find q solutions with the greedy algorithm
        # Compute EI_D for all the points in Q
        pop_size = self.ref_vecs.shape[0]
        EIDs = torch.zeros(pop_size,pop_size).to(**tkwargs)
        for i in range(pop_size):
            temp_mean = candidate_mean[i:i+1].repeat(pop_size,1)
            temp_std = candidata_std[i:i+1].repeat(pop_size,1)
            EIDs[i, :] = self._get_acquisition(temp_mean, temp_std, dir_vecs, xis)
 
        Qb = []
        temp = EIDs.clone()
        beta = torch.zeros(pop_size).to(**tkwargs)
        for i in range(batch_size):
            index = torch.argmax(torch.sum(temp, dim=1))
            Qb.append(index.item())
            beta = beta + temp[index, :]
            # Update temp: [EI_D(x|\lambda) - beta]_+
            temp = EIDs - beta[None, :].repeat(pop_size, 1)
            temp[temp < 0] = 0
        
        # observe new values
        new_x = candidate_x[Qb]
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
        self.archive_y_nds = self.archive_y[self.FrontNo[0]].clone()
        # minimization problem
        self.hv_list = np.append(self.hv_list,[[self.HV.compute(-self.archive_y_nds)]],axis=0)
        self.FE = np.append(self.FE,[[self.archive_x.shape[0]]],axis=0)
        
        if self.debug:
            self.plot_objs()
      
  

    def _moead_gr(self, ref_vecs: Tensor, pref_incs: Tensor)->Tuple[Tensor,Tensor,Tensor]:
        # using MOEA/D-GR to solve subproblems
        maxIter = 100
        pop_size = self.ref_vecs.shape[0] # pop_size
        T = int(np.ceil(0.1 * pop_size).item())  # size of neighbourhood: 0.1*N
        B = torch.argsort(torch.cdist(ref_vecs, ref_vecs), dim=1)[:, :T]
 

        # the initial population for MOEA/D
        x_ini = torch.from_numpy(lhs(self.n_dim,samples=pop_size)).to(**tkwargs)
        pop_x = (self._upper_x-self._lower_x)*x_ini + self._lower_x
        # gp poterior
        pop_mean, pop_std = self.GaussianProcess.evaluate(pop_x, calc_std=True, calc_gradient=False) 
        
        # calculate the values of preference conditional acquisition functions
        pop_acq = self._get_acquisition(pop_mean, pop_std, ref_vecs, pref_incs)

        # optimization
        for gen in range(maxIter - 1):
            for i in range(pop_size):
                if torch.rand(1) < 0.8:  # delta
                    P = B[i, np.random.permutation(B.shape[1])]
                else:
                    P = np.random.permutation(pop_size)
                # generate an offspring 1*d
                off_x = self._operator_DE(pop_x[i:i+1, :], pop_x[P[0:1], :], pop_x[P[1:2], :])
                 
                off_mean, off_std = self.GaussianProcess.evaluate(off_x, calc_std=True, calc_gradient=False) 
                
                # Global Replacement  MOEA/D-GR
                # Find the most approprite subproblem and its neighbourhood
                acq_all = self._get_acquisition(off_mean.repeat(pop_size,1), off_std.repeat(pop_size,1),ref_vecs, pref_incs)
                best_index = torch.argmax(acq_all) 
                P = B[best_index, :]  # replacement neighborhood

                offindex = P[pop_acq[P] < acq_all[P]]
                if len(offindex) > 0:
                    pop_x[offindex, :] = off_x.repeat(len(offindex), 1)
                    pop_mean[offindex, :] = off_mean.repeat(len(offindex), 1)
                    pop_std[offindex, :] = off_std.repeat(len(offindex), 1)
                    pop_acq[offindex] = acq_all[offindex]

        return pop_x,pop_mean, pop_std

    def _operator_DE(self, Parent1, Parent2, Parent3):
        '''
            generate one offspring by P1 + 0.5*(P2-P3) and polynomial mutation.
        '''
        # Parameter
        CR = 1
        F = 0.5
        proM = 1
        disM = 20
        #
        N, D = Parent1.shape
        # Differental evolution
        Site = torch.rand(N, D) < CR
        Offspring = Parent1.clone()
        Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])
        # Polynomial mutation
        Lower = self._lower_x
        Upper = self._upper_x

        U_L = Upper - Lower
        Site = torch.rand(N, D).to(**tkwargs) < proM / D
        mu = torch.rand(N, D).to(**tkwargs)
        temp = torch.logical_and(Site, mu <= 0.5)
        Offspring = torch.min(torch.max(Offspring, Lower), Upper)
     
        delta1 = (Offspring - Lower) / U_L
        delta2 = (Upper - Offspring) / U_L
        #  mu <= 0.5
        val = 2. * mu + (1 - 2. * mu) * ((1. - delta1).pow(disM + 1))
        Offspring[temp] = Offspring[temp] + ((val[temp]).pow(1.0 / (disM + 1)) - 1.) * U_L[temp]
        # mu > 0.5
        temp = torch.logical_and(Site, mu > 0.5)
        val = 2. * (1.0 - mu) + 2. * (mu - 0.5) * ((1. - delta2).pow(disM + 1))
        Offspring[temp] = Offspring[temp] + (1.0 - (val[temp]).pow(1.0 / (disM + 1))) * U_L[temp]
    
        return Offspring
    
    def plot_objs(self):
        fig = plt.figure()
        archive_y_nds = self.archive_y_nds.to('cpu')
        plt.scatter(archive_y_nds[...,0], archive_y_nds[...,1], label=self.solver_name)
        if hasattr(self.problem, '_get_pf'):
            plt.plot(self.problem._get_pf()[:,0], self.problem._get_pf()[:,1], label='PF')

        plt.legend(fontsize=16)
        plt.xlabel('$f_1$', fontsize=18)
        plt.ylabel('$f_2$', fontsize=18)
        plt.show()
    
if __name__ == '__main__':
    import time
    from utils.lhs import lhs
    import matplotlib.pyplot as plt
    from test_functions import ZDT1,ZDT2,ZDT3,ZDT6
    
    # minimization
    problem = ZDT1(n_obj=2,n_dim=8)
    n_init = 11*problem.n_dim-1
    batch_size = 5
    maxFE = 200
    ts = time.time()
 
    x_init = torch.from_numpy(lhs(problem.n_dim, samples=n_init)).to(**tkwargs)
    y_init = problem.evaluate(x_init)
    solver = DirHVEGOSolver(problem, maxFE, batch_size, x_init, y_init)
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
 