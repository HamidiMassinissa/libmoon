import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from .base import SurrogateModel
import torch
device = 'cpu'

def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


class GaussianProcess(SurrogateModel):
    
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    
    '''
    Gaussian process
    '''
    def __init__(self, n_obj, n_var, **kwargs):
        super().__init__(n_obj, n_var)
        nu = 5
        self.nu = nu
        self.gps = []

        def constrained_optimization(obj_func, initial_theta, bounds):
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
            '''
            NOTE: Temporarily disable the checking below because this error sometimes occurs:
                ConvergenceWarning: lbfgs failed to converge (status=2):
                ABNORMAL_TERMINATION_IN_LNSRCH
                , though we already optimized enough number of iterations and scaled the data.
                Still don't know the exact reason of this yet.
            '''
            # _check_optimize_result("lbfgs", opt_res)
            return opt_res.x, opt_res.fun

        for _ in range(n_obj):
            if nu > 0:
                main_kernel = Matern(length_scale=np.ones(n_var), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)), nu=0.5 * nu)
            else:
                main_kernel = RBF(length_scale=np.ones(n_var), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))
            
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-3), np.sqrt(1e3))) * \
                main_kernel + \
                ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))
            
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=constrained_optimization)
            self.gps.append(gp)

    def fit(self, X, Y):
        for i, gp in enumerate(self.gps):
            gp.fit(X, Y[:, i])
        
    def evaluate(self, X, calc_std=False, calc_gradient=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std

        for gp in self.gps:

            # mean
            K = gp.kernel_(X, gp.X_train_) # K: shape (N, N_train)
            y_mean = K.dot(gp.alpha_)
            
            F.append(y_mean) # y_mean: shape (N,)

            if calc_std:
                
                L_inv = solve_triangular(gp.L_.T,
                                                np.eye(gp.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)

                y_var = gp.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                    np.dot(K, K_inv), K)

                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    y_var[y_var_negative] = 0.0

                y_std = np.sqrt(y_var)

                S.append(y_std) # y_std: shape (N,)

            if not (calc_gradient): continue

            ell = np.exp(gp.kernel_.theta[1:-1]) # ell: shape (n_var,)
            sf2 = np.exp(gp.kernel_.theta[0]) # sf2: shape (1,)
            d = np.expand_dims(cdist(X / ell, gp.X_train_ / ell), 2) # d: shape (N, N_train, 1)
            X_, X_train_ = np.expand_dims(X, 1), np.expand_dims(gp.X_train_, 0)
            dd_N = X_ - X_train_ # numerator
            dd_D = d * ell ** 2 # denominator
            dd = safe_divide(dd_N, dd_D) # dd: shape (N, N_train, n_var)

            if calc_gradient:
                if self.nu == 1:
                    dK = -sf2 * np.exp(-d) * dd

                elif self.nu == 3:
                    dK = -3 * sf2 * np.exp(-np.sqrt(3) * d) * d * dd

                elif self.nu == 5:
                    dK = -5. / 3 * sf2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd

                else: # RBF
                    dK = -sf2 * np.exp(-0.5 * d ** 2) * d * dd

                dK_T = dK.transpose(0, 2, 1) # dK: shape (N, N_train, n_var), dK_T: shape (N, n_var, N_train)
                
            if calc_gradient:
                dy_mean = dK_T @ gp.alpha_ # gp.alpha_: shape (N_train,)
                dF.append(dy_mean) # dy_mean: shape (N, n_var)

                # TODO: check
                if calc_std:
                    K = np.expand_dims(K, 1) # K: shape (N, 1, N_train)
                    K_Ki = K @ K_inv # gp._K_inv: shape (N_train, N_train), K_Ki: shape (N, 1, N_train)
                    dK_Ki = dK_T @ K_inv # dK_Ki: shape (N, n_var, N_train)

                    dy_var = -np.sum(dK_Ki * K + K_Ki * dK_T, axis=2) # dy_var: shape (N, n_var)
                    #print(dy_var.shape)
                    #print(np.expand_dims(y_std,1).shape)
                    #dy_std = 0.5 * safe_divide(dy_var, y_std) # dy_std: shape (N, n_var)
                    if np.min(y_std) != 0:
                        dy_std = 0.5 * dy_var / np.expand_dims(y_std,1) # dy_std: shape (N, n_var)
                    else:
                        dy_std=np.zeros(dy_var.shape)
                    dS.append(dy_std)
 
        F = torch.from_numpy(np.stack(F, axis=1)).to(device) 
        if not calc_std:
            return F 
        
        S = torch.from_numpy(np.stack(S, axis=1)).to(device) 
        if not calc_gradient: 
            return F, S 
        
        dF = torch.from_numpy(np.stack(dF, axis=1)).to(device) 
        dS = torch.from_numpy(np.stack(dS, axis=1)).to(device) 
        return F, S, dF, dS
        
        
