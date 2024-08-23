from environment import *
from algorithm import *
from data.data import *
from utils import *
import time

# get environment, utilities, and policies
M, utilities, policies = construct_env_utilities_and_policies()

# choose number of trajectories
n_traj = 10000
n_est = 5

all_comp_abs = []
all_comp_rel = []

# loop over all participants
for i, piE in enumerate(policies):
    print('Participant '+str(i))
    start_time = time.time()

    # make experiment reproducible
    np.random.seed(0)

    # estimate expert's return distribution
    etaE = M.estimate_return_distribution(pi=piE, n_traj=n_traj)

    comp_abs = {}
    comp_rel = {}

    cs_abs = []
    cs_rel = []
    for seed in range(n_est):
        np.random.seed(seed)
        # estimate compatibility with SG utility
        c_abs, c_rel = caty(
            M=M,
            etaE=etaE,
            U=utilities[i],  # SG utility of participant i
            n_traj=n_traj
        )
        cs_abs.append(c_abs)
        cs_rel.append(c_rel)
    
    comp_abs['SG_mean'] = np.mean(cs_abs)
    comp_abs['SG_std'] = np.std(cs_abs)
    comp_rel['SG_mean'] = np.mean(cs_rel)
    comp_rel['SG_std'] = np.std(cs_rel)


    # estimate compatibility with all other utilities
    for ut in ['sqrt', 'square', 'linear']:
        cs_abs = []
        cs_rel = []
        for seed in range(n_est):
            np.random.seed(seed)
            c_abs, c_rel = caty(
                M=M,
                etaE=etaE,
                U=get_utility(ut, M.M.H,M.d),
                n_traj=n_traj
            )
            cs_abs.append(c_abs)
            cs_rel.append(c_rel)
        comp_abs[ut+'_mean'] = np.mean(cs_abs)
        comp_abs[ut+'_std'] = np.std(cs_abs)
        comp_rel[ut+'_mean'] = np.mean(cs_rel)
        comp_rel[ut+'_std'] = np.std(cs_rel)
    
    # append
    all_comp_abs.append(comp_abs)
    all_comp_rel.append(comp_rel)

    end_time = time.time()
    print('-'*10+' '+'{:.3}'.format((end_time - start_time)/60)+'min')

# save results
np.save('results/exp1/abs_comp.npy', all_comp_abs)
np.save('results/exp1/rel_comp.npy', all_comp_rel)