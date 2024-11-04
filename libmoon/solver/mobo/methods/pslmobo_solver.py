"""
Reference:
    Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang. Pareto Set Learning for
    Expensive Multiobjective Optimization. Advances in Neural Information Processing
    Systems (NeurIPS) , 2022.
    
"""

import torch
from torch import Tensor
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.hypervolume import Hypervolume

import os
import os.path
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.BPSModel import BayesianPSModel as ParetoSetModel
from methods.base_solver_BPSL import BayesianPSL

class PSLMOBOSolver(BayesianPSL):
    def __init__(self, problem, MAX_FE: int, BATCH_SIZE: int, x_init: Tensor, y_init: Tensor):
        super().__init__(problem, MAX_FE, BATCH_SIZE, x_init, y_init)
        self.solver_name = 'pslmobo'
        # Parameter Setting
        self.n_steps = 1000 # number of learning steps
        self.n_pref_update = 10 # number of sampled preferences per step
        self.coef_lcb = 0.5 # coefficient of LCB
        self.n_candidate = 1000  # number of sampled candidates on the approxiamte PF
        self.learning_rate = 1e-3

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
  
            # get the current coressponding solutions
            x = self.psmodel(pref_vec)
            mean, std, mean_grad, std_grad = self.GPModelList.evaluate(x, calc_std=True, calc_gradient=True) 

            # calculate the value/grad of tch decomposition with LCB
            value = mean - self.coef_lcb * std    # n_pref_update *  n_obj  
            # n_pref_update *  n_obj * n_var   
            value_grad = mean_grad - self.coef_lcb * std_grad
            tch_idx = torch.argmax((pref_vec) * (value - self.z), axis = 1)
            tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
            # n_pref_update *  n_var  
            tch_grad = (pref_vec)[tch_idx_mat].view(self.n_pref_update,1) *  value_grad[tch_idx_mat] #+ 0.01 * torch.sum(value_grad, axis = 1) 
            tch_grad = tch_grad / torch.norm(tch_grad, dim = 1, keepdim=True)
            # gradient-based pareto set model update
            optimizer.zero_grad()
            self.psmodel(pref_vec).backward(tch_grad)
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
  
        Y_candidate = candidate_mean - self.coef_lcb * candidata_std 
        # hv
        ref_point = torch.max(torch.cat((self.train_y_nds,Y_candidate),dim=0),axis=0).values.data
        hv = Hypervolume(ref_point=-ref_point) # minimization problem
        # greedy batch selection 
        best_subset_list = []
        Y_p = self.train_y_nds.clone()
        for b in range(batch_size):        
            best_hv_value = 0
            best_subset = None
            for k in range(self.n_candidate):
                Y_comb = torch.cat((Y_p,Y_candidate[k:k+1,:]),dim=0)
                hv_value_subset = hv.compute(-Y_comb) # minimization problem
                if hv_value_subset > best_hv_value:
                    best_hv_value = hv_value_subset
                    best_subset = k
                    
            Y_p = torch.cat((Y_p,Y_candidate[best_subset:best_subset+1,:]),dim=0) 
            best_subset_list.append(best_subset)  
       
        # evaluate the selected n_sample solutions
        new_x = candidate_x[best_subset_list]
        return new_x
    
if __name__ == '__main__':
    import time
    from utils.lhs import lhs
    import matplotlib.pyplot as plt
    from test_functions import ZDT1,ZDT2,ZDT3,ZDT6
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    # minimization
    problem = ZDT1(n_obj=2,n_dim=8)
    n_init = 11*problem.n_dim-1
    batch_size = 5
    maxFE = 200
    ts = time.time()
 
    x_init = torch.from_numpy(lhs(problem.n_dim, samples=n_init)).to(**tkwargs)
    y_init = problem.evaluate(x_init)
    solver = PSLMOBOSolver(problem, maxFE, batch_size, x_init, y_init)
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
  
 
    
    
   
    
 