from environment import *
from algorithm import *
from data.data import *
from utils import *
import time

# get environment, utilities, and policies
M, utilities, policies = construct_env_utilities_and_policies()

############### hyperparameters

# choose participant to analyse
participant = 9

# choose number trajectories for expert's return distribution
n_traj = 10000
n_est = 5
T = 70
L = 10

folder = 'results/exp2/'

# choose learning rate for TRACTOR
alphas = [1e-2, 5e-1, 5, 100, 1000, 10000]

############### execute

print('START')

# set seed for reproducibility
np.random.seed(0)

# estimate expert's return distribution
piE = policies[participant]
etaE = M.estimate_return_distribution(pi=piE, n_traj=n_traj)

# loop for every initial utility
for ut in ['sqrt', 'square', 'linear']:
    print('*'*10 + ' UTILITY: '+ut)

    initial_U = get_utility(ut,M.M.H,M.d)

    # loop over learning rates
    for alpha in alphas:

        # loop for every seed
        for seed in range(n_est):
            np.random.seed(seed)

            start_time = time.time()

            U, u_list, subopts = tractor(
                etaE=etaE,
                T=T,
                alpha=alpha,
                M=M,
                n_traj=n_traj,
                initial_U=np.copy(initial_U),
                L=L,
                use_lipschitz=True,  # L-Lipschitz constraint
                adam=False,
                verbose=True
            )

            # save results to file
            np.save(folder+'U_L_'+ut+'_'+str(alpha)+'_'+str(seed)+'.npy', U)
            np.save(folder+'u_list_L_'+ut+'_'+str(alpha)+'_'+str(seed)+'.npy', u_list)
            np.save(folder+'subopts_L_'+ut+'_'+str(alpha)+'_'+str(seed)+'.npy', subopts)

            end_time = time.time()
            print('-'*10+' '+'{:.3}'.format((end_time - start_time)/60)+'min')