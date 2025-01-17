import numpy as np
import torch
from libmoon.problem.synthetic.mop import BaseMOP

class F1(BaseMOP):
    def __init__(self,
                n_var: int,
                n_obj: int=None,
                lbound: np.ndarray=None,
                ubound: np.ndarray=None,
                n_cons: int = 0,
                ) -> None:
        
        self.n_var = n_var
        self.n_obj = 2
        self.lbound = torch.zeros(n_var).float()
        self.ubound = torch.ones(n_var).float()
        self.problem_name = "F1"

    def _evaluate_torch(self, x):
        n = x.shape[1]

        sum1 = sum2 =  0.0
        count1 = count2 =  0.0

        for i in range(2,n+1):
            yi = x[:,i-1] - torch.pow(2 * x[:,0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 )))

        objs = torch.stack([f1,f2]).T

        return objs
    
# import necessary modules
if __name__ == '__main__':
    from libmoon.util.synthetic import synthetic_init
    from libmoon.util.prefs import get_uniform_pref
    from libmoon.util.problems import get_problem
    from libmoon.solver.gradient.methods import EPOSolver

    problem = get_problem(problem_name='VLMOP2')
    prefs = get_uniform_pref(n_prob=5, n_obj=problem.n_obj, clip_eps=1e-2)
    solver = EPOSolver(step_size=1e-2, n_iter=1000, tol=1e-2, problem=problem, prefs=prefs)
    res = solver.solve(x=synthetic_init(problem, prefs))
    print(res['x'])
