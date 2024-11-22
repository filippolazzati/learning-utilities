import cvxpy as cp
import numpy as np
from environment import *
from typing import Tuple

def caty(
        M: DiscretizedMDP,
        etaE: np.array,
        U: np.array,
        n_traj: int
    ) -> Tuple[float, float]:
    """
    Implementation of caty-UL algorithm analogously to the description
    provided in the paper, but with some slight differences due to the
    different setting.

    Input arguments are:
    - M: discretized MDP to consider
    - etaE: return distribution of the expert's policy
    - U: utility whose (non)compatibility has to be computed
    - n_traj: number of trajectories used to estimate the return distributions

    Given a utility function U and a distribution over returns etaE in
    a certain environment M, caty computes and returns an absolute and
    relative estimates of (non)compatibility of input utility U with etaE
    in M.
    """

    # get expert's policy expected utility
    JE = np.dot(U, etaE)

    # get optimal policy return distribution
    pi = M.compute_optimal_policy(U)
    eta = M.estimate_return_distribution(pi=pi, n_traj=n_traj)
    J = np.dot(U, eta)

    # estimate compatibility
    comp_abs = J - JE  # absolute
    comp_rel = comp_abs / J  # relative

    return comp_abs, comp_rel

def tractor(
        etaE: np.ndarray,
        T: int,
        alpha: float,
        M: DiscretizedMDP,
        n_traj: int,
        initial_U: np.ndarray,
        L: float,
        use_lipschitz: bool,
        epsilon: float = 1e-5,
        adam: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, list, list]:
    """
    Implementation of tractor-UL algorithm described in the paper, with
    both adam and Gradient Descent (GD) variants.

    Input arguments are:
    - etaE: return distribution of the expert's policy
    - T: maximum number of iterations of the algorithm
    - alpha: learning rate for both GD and adam
    - M: discretized MDP to consider
    - n_traj: number of trajectories used to estimate the return distributions
    - initial_U: initial utility considered by the algorithm
    - L: Lipschitz constant to use to enforce the utilities to be L-Lipschitz
    - use_lipschitz: whether to enforce L-Lipschitz constraint
    - epsilon: if the (non)compatibility of utility is smaller than epsilon,
    terminate
    - adam: whether to use adam instead of GD
    - beta1, beta2, eps_adam: parameters for adam implementation
    - verbose: if True, print all steps of algorithm

    tractor returns the utility function found by the algorithm, a list with
    all the utilities (non averaged) computed at the various iterations, and a
    list with the suboptimalities of such utilities.
    """

    U_list = []
    U = initial_U
    U_list.append(np.copy(U))

    suboptimalities = []

    # adam
    if adam:
        # initialize first and second moments
        m = [0.0 for _ in range(len(U))]
        v = [0.0 for _ in range(len(U))]

    # loop
    for t in range(T):
        # compute candidate utility
        U_avg = np.mean(np.stack(U_list, axis=0), axis=0)

        # find optimal return distribution for U_avg
        pi_avg = M.compute_optimal_policy(U_avg)
        eta_avg = M.estimate_return_distribution(pi=pi_avg, n_traj=n_traj)

        # check termination condition -> use U_avg
        subopt = np.dot(U_avg, eta_avg-etaE)
        suboptimalities.append(subopt)

        # print iteration
        if verbose:
            print('Iteration ', t,', expert suboptimality U(eta-etaE)= ',subopt)

        if subopt <= epsilon:
            print('Terminate because subopt < '+str(epsilon))
            break

        # compute optimal policy return distribution
        pi = M.compute_optimal_policy(U)
        eta = M.estimate_return_distribution(pi=pi, n_traj=n_traj)

        # compute the gradient
        g = eta - etaE

        # adam
        if adam:
            # build a solution one variable at a time
            for i in range(len(U)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
                mhat = m[i] / (1.0 - beta1**(t+1))
                vhat = v[i] / (1.0 - beta2**(t+1))
                U[i] = U[i] - alpha * mhat / (np.sqrt(vhat) + eps_adam)
        else:
            U = U - alpha*g

        # project U onto the feasible set
        U = project(U=U, H=M.M.H, L=L, eps0=M.eps0, use_lipschitz=use_lipschitz)

        U_list.append(np.copy(U))

    return U_avg, U_list, suboptimalities

def tractor_many_envs(
        N: int,
        etaEs: List[np.ndarray],
        Ms: List[DiscretizedMDP],
        T: int,
        alpha: float,
        n_traj: int,
        initial_U: np.ndarray,
        L: float,
        use_lipschitz: bool,
        epsilon: float = 1e-2,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, list, list]:
    """
    Analogous version of tractor with demonstrations in N>=1 environments.
    """

    U_list = []
    U = initial_U
    U_list.append(np.copy(U))

    suboptimalities = []

    # loop
    for t in range(T):
        if t==0 or (t+1)%10 ==0:
            # compute candidate utility
            U_avg = np.mean(np.stack(U_list, axis=0), axis=0)

            max_comp = -np.infty
            for i in range(N):
                # find compatibility for U_avg
                pi_avg = Ms[i].compute_optimal_policy(U_avg)
                eta_avg = Ms[i].estimate_return_distribution(pi=pi_avg, n_traj=n_traj)
                C = np.dot(U_avg, eta_avg-etaEs[i])
                max_comp = np.maximum(max_comp, C)

            # check termination condition -> use U_avg
            suboptimalities.append(max_comp)

            # print iteration
            if verbose:
                print('Iteration ', t,', expert suboptimality max_i C_i= ',max_comp)

            if max_comp <= epsilon:
                print('Terminate because max_comp < '+str(epsilon))
                break

        # compute the gradient
        g = 0
        for i in range(N):
            # find return distribution of optimal policy for U
            pi = Ms[i].compute_optimal_policy(U)
            eta = Ms[i].estimate_return_distribution(pi=pi, n_traj=n_traj)
            g += eta - etaEs[i]
        
        # apply the gradient
        U = U - alpha*g

        # project U onto the feasible set
        U = project(U=U, H=Ms[0].M.H, L=L, eps0=Ms[0].eps0, use_lipschitz=use_lipschitz)

        U_list.append(np.copy(U))

    return U_avg, U_list, suboptimalities

def project(
        U: np.ndarray,
        H: int,
        L: float,
        eps0: float,
        use_lipschitz: bool
    ) -> np.ndarray:
    """
    This function projects  the input utility U onto the space of bounded
    non-decreasing utility functions. L represents the Lipschitz constant
    for the continuity of the utility function. This function is used by
    tractor.

    Input arguments are:
    - U: the utility function to project
    - H: the horizon of the discretized MDP
    - L: Lipschitz constant to use to enforce the utilities to be L-Lipschitz
    - eps0: discretization parameter for the discretized MDP
    - use_lipschitz: whether to enforce L-Lipschitz constraint

    This function returns the projected utility.
    """
    d = len(U)
    U_new = cp.Variable(d)
    objective = cp.Minimize(cp.norm(U_new - U, 2))

    # constraints <= H
    A1 = np.eye(d)
    b1 = H*np.ones(d)

    # constraints >= 0
    A2 = -np.eye(d)
    b2 = np.zeros(d)

    # U1=0, UH=H
    b1[0] = 0
    b2[-1] = -H

    # constraints Ui<=Ui+1
    A3 = np.zeros((d-1, d))
    for i in range(d-1):
        A3[i,i] = 1
        A3[i,i+1] = -1
    b3 = np.zeros(d-1)

    # Lipschitz constraints
    if use_lipschitz:
        A4 = np.zeros((((d-1)*d)//2, d))
        b4 = L*eps0*np.ones(((d-1)*d)//2)
        idx = 0
        for i in range(d-1):
            for j in range(i+1,d):
                A4[idx,i] = -1
                A4[idx,j] = +1
                b4[idx] *= (j-i)
                idx += 1

        # concatenate
        A = np.concatenate((A1,A2,A3,A4))
        b = np.concatenate((b1,b2,b3,b4))
    
    else:
        # concatenate
        A = np.concatenate((A1,A2,A3))
        b = np.concatenate((b1,b2,b3))

    # solve
    constraints = [A @ U_new <= b]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # get the projected vector
    U_projected = U_new.value

    return U_projected
