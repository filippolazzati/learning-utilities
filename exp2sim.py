from data.data import *
from utils import *
from environment import *
from algorithm import *
import numpy as np

#################### change S and A

# S, A, H, eps_0
exps = {
    # 2: [20, 5, 5, 1e-2],
    # 3: [100, 10, 5, 1e-2],
    4: [1000, 20, 5, 1e-2],
}
n_traj = 10000
n_seeds = 1
T = 70
L = 10
alpha = 5
N = 1

for exp, (S,A,H,eps_0) in exps.items():
    print('exp: '+str(exp))

    Ms, piEs = construct_random_envs_and_policies(
        N=N,
        S=S,
        A=A,
        H=H,
        eps_0=eps_0,
        seed=1
    )

    initial_U = get_utility('linear', H, Ms[0].d)
    UU = get_utility('convex-concave', H, Ms[0].d)
    etaEs = []

    for i, M in enumerate(Ms):
        piE = M.compute_optimal_policy(UU)
        piE = inject_noise(piE)
        etaE = M.estimate_return_distribution(pi=piE, n_traj=n_traj)
        etaEs.append(etaE)

    # loop for every seed
    for seed in range(n_seeds):
        np.random.seed(seed)

        U, u_list, subopts = tractor_many_envs(
            N=N,
            etaEs=etaEs,
            Ms=Ms,
            T=T,
            alpha=alpha,
            n_traj=n_traj,
            initial_U=np.copy(initial_U),
            L=L,
            use_lipschitz=True,  # L-Lipschitz constraint
            verbose=True
        )

        # save results to file
        folder = 'results/exp2sim/S_'+str(S)+'_A_'+str(A)+'/'
        np.save(folder+'U_L_'+str(seed)+'.npy', U)
        np.save(folder+'u_list_L_'+str(seed)+'.npy', u_list)
        np.save(folder+'subopts_L_'+str(seed)+'.npy', subopts)


#################### N = 5

N = 5
S = 4
A = 3
H = 5

Ms, _ = construct_random_envs_and_policies(
    N=N,
    S=S,
    A=A,
    H=H,
    eps_0=1e-2,
    seed=0
)

UU = get_utility('convex-concave', H, Ms[0].d)

initial_U = get_utility('linear', H, Ms[0].d)
n_traj = 10000

etaEs = []

for i,M in enumerate(Ms):
    if i%5==0:
        print(i)
    piE = M.compute_optimal_policy(UU)
    piE = inject_noise(piE)
    etaE = M.estimate_return_distribution(pi=piE, n_traj=n_traj)
    etaEs.append(etaE)

n_seeds = 1
T = 70
L = 10
alpha = 1

# loop for every seed
for seed in range(n_seeds):
    np.random.seed(seed)

    U, u_list, subopts = tractor_many_envs(
        N=N,
        etaEs=etaEs,
        Ms=Ms,
        T=T,
        alpha=alpha,
        n_traj=n_traj,
        initial_U=np.copy(initial_U),
        L=L,
        use_lipschitz=True,  # L-Lipschitz constraint
        verbose=True
    )

    # save results to file
    folder = 'results/exp2sim/N5/'
    np.save(folder+'U_L_'+str(seed)+'.npy', U)
    np.save(folder+'u_list_L_'+str(seed)+'.npy', u_list)
    np.save(folder+'subopts_L_'+str(seed)+'.npy', subopts)


#################### N = 20

N = 20
S = 4
A = 3
H = 5

Ms, _ = construct_random_envs_and_policies(
    N=N,
    S=S,
    A=A,
    H=H,
    eps_0=1e-2,
    seed=0
)

UU = get_utility('convex-concave', H, Ms[0].d)

initial_U = get_utility('linear', H, Ms[0].d)
n_traj = 10000

etaEs = []

for i,M in enumerate(Ms):
    if i%5==0:
        print(i)
    piE = M.compute_optimal_policy(UU)
    piE = inject_noise(piE)
    etaE = M.estimate_return_distribution(pi=piE, n_traj=n_traj)
    etaEs.append(etaE)

n_seeds = 1
T = 70
L = 10
alpha = 1

# loop for every seed
for seed in range(n_seeds):
    np.random.seed(seed)

    U, u_list, subopts = tractor_many_envs(
        N=N,
        etaEs=etaEs,
        Ms=Ms,
        T=T,
        alpha=alpha,
        n_traj=n_traj,
        initial_U=np.copy(initial_U),
        L=L,
        use_lipschitz=True,  # L-Lipschitz constraint
        verbose=True
    )

    # save results to file
    folder = 'results/exp2sim/N20/'
    np.save(folder+'U_L_'+str(seed)+'.npy', U)
    np.save(folder+'u_list_L_'+str(seed)+'.npy', u_list)
    np.save(folder+'subopts_L_'+str(seed)+'.npy', subopts)