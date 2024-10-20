import torch
import math
import numpy as np  
from utils import lhs

from surrogate_models import  GPModelList

from matplotlib import pyplot as plt

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

#################################################################################
def plot_gp_m2d1(name : str, train_x: np.ndarray,train_y: np.ndarray,test_x: np.ndarray,
                 test_y: np.ndarray, mean: np.ndarray, std: np.ndarray):
    # train_x: n*d, train_y: n*m, 
    # test_x: N*d, mean: N*m, std: N:m
    upper = mean + std
    lower = mean - std
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    plt.rcParams['font.sans-serif']=['Times New Roman']
    plt.rcParams['axes.unicode_minus']=False
    # Plot training data as black stars
    y1_ax.plot(train_x, train_y[:, 0], 'k*', label="observed samples")
    # Predictive mean as blue line
    y1_ax.plot(test_x, mean[:, 0], color='blue', linewidth=2.0, linestyle="-", label='$\mu_1(x)$')
    y1_ax.plot(test_x, test_y[:,0], color='black', linewidth=2.0, linestyle="--", label='$f_1(x)=\sin(2\pi x)$')
    # Shade in confidence
    y1_ax.fill_between(test_x[:,0], lower[:, 0], upper[:, 0], alpha=0.5, label='$\mu_1(x)\pm \sigma_1(x)$')
    # y1_ax.set_ylim([-3, 3])
    y1_ax.legend(loc='lower left')
    y1_ax.set_title('Observed Values (Likelihood)')
    
    # Plot training data as black stars
    y2_ax.plot(train_x, train_y[:, 1], 'k*', label="observed samples")
    # Predictive mean as blue line
    y2_ax.plot(test_x, mean[:, 1], color='blue', linewidth=2.0, linestyle="-", label='$\mu_2(x)$')
    y2_ax.plot(test_x, test_y[:,1], color='black', linewidth=2.0, linestyle="--", label='$f_2(x)=\cos(2\pi x)$')
    # Shade in confidence
    y2_ax.fill_between(test_x[:,0], lower[:, 1], upper[:, 1], alpha=0.5, label='$\mu_2(x)\pm \sigma_2(x)$')
    # y2_ax.set_ylim([-3, 3])
    y2_ax.legend(loc='lower left')
    y2_ax.set_title('Observed Values (Likelihood)')
    figure_name =  './Data/'+name+'.pdf'
    plt.xticks(fontsize=13)

    plt.yticks(fontsize=13)
    # plt.savefig(figure_name, dpi=200)
################################################################################# 
n_init, n_test = 5, 100
n_var, n_obj = 1, 2
 

train_x = torch.from_numpy(lhs(n_var, samples=n_init)).to(**tkwargs) 
 
train_y = torch.stack([
    torch.sin(12*train_x -4)*torch.pow(6*train_x-2,2),
    torch.cos(train_x * (2 * math.pi)), 
])[:,:,0].T
 
test_x = torch.linspace(0, 1, n_test).reshape(n_test, n_var)
 
test_x.requires_grad = True
test_y = torch.stack([
    torch.sin(12*test_x -4)*torch.pow(6*test_x-2,2),
    torch.cos(test_x * (2 * math.pi)),
])[:,:,0].T

train_x_np = train_x.detach().cpu().numpy()
train_y_np = train_y.detach().cpu().numpy()
test_x_np = test_x.detach().cpu().numpy()
true_y_np =  test_y.detach().cpu().numpy()
################################# BoTorch ##################################### 
# gp_models =  GPSingleTask(n_var,n_obj, **tkwargs)
# gp_models.fit(train_x, train_y) 
 
# mean, std, mean_grad, std_grad = gp_models.evaluate(test_x, cal_std=True, cal_grad=True) 
# plot_gp_m2d1('GPSingleTask', train_x_np, train_y_np, test_x_np,true_y_np,
#               mean.detach().cpu().numpy(), std.detach().cpu().numpy()) 
 
################################# BoTorch GPModelList##################################### 
gp_botorch =  GPModelList(n_var,n_obj, **tkwargs)
gp_botorch.fit(train_x, train_y) 
 
mean_bo, std_bo, mean_grad_bo, std_grad_bo = gp_botorch.evaluate(test_x, calc_std=True, calc_gradient=True) 
plot_gp_m2d1('GPModelList', train_x_np, train_y_np, test_x_np,true_y_np,
              mean_bo.detach().cpu().numpy(), std_bo.detach().cpu().numpy()) 

################################ Sklearn ######################################  
#gp_skl =  GaussianProcess(n_var, n_obj)
# gp_skl =  GaussianProcess(n_var, n_obj)
# gp_skl.fit(train_x_np, train_y_np) 
 
 
# mean_skl, std_skl, mean_grad_skl, std_grad_skl = gp_skl.evaluate(test_x_np, calc_std=True, calc_gradient=True) 
# plot_gp_m2d1('sklearn',train_x_np, train_y_np, test_x_np,true_y_np,
#              mean_skl.detach().cpu().numpy(), std_skl.detach().cpu().numpy()) 
################################# pySMT ####################################### 
# gp_smt =  GPsmt(n_var, n_obj)
# gp_smt.fit(train_x_np, train_y_np) 

# mean_smt, std_smt, mean_grad_smt, std_grad_smt = gp_smt.evaluate(test_x_np, cal_std=True, cal_grad=True) 
# plot_gp_m2d1('GPsmt',train_x_np, train_y_np, test_x_np,true_y_np,
              # mean_smt, std_smt)

# 1. pySMT只适合用rbf kernel
# 2. rbf kernel: pySMT 和 sklearn的结果很类似, 包括梯度结果, botorch的结果也差不多
# 3. matern52： GPtorch 和 sklearn结果更类似
