from typing import List
import numpy as np

class MDP:
    """
    This class models the notion of tabular episodic finite-horizon MDP with a
    single initial state.
    """
    S: int
    A: int
    H: int
    r: np.ndarray  # s,a,h
    p: np.ndarray  # s,a,h,s'
    s_0: int

    def __init__(
            self,
            S: int,
            A: int,
            H: int,
            r: np.ndarray,
            p: np.ndarray,
            s_0: int
        ):
        """
        Constructor method for class MDP.

        Input arguments:
        - S: an integer representing the cardinality of the state space
        - A: an integer representing the cardinality of the action space
        - H: the horizon
        - r: the reward function, represented as a SxAxH np.ndarray
        - p: the transition model, represented as a SxAxHxS np.ndarray
        - s_0: the initial state, an integer in {0,1,...,S-1}
        """
        if H < 2:
            raise Exception("Sorry, H must be at least 2")
        self.S = S
        self.A = A
        self.H = H
        self.r = r
        self.p = p
        self.s_0 = s_0
    
class DiscretizedMDP:
    """
    This class models the notion of discretized MDP as described in the paper.
    """
    M: MDP
    r: np.ndarray # discretized, s,a,h
    eps0: float  # covering radius
    y_values: List[np.ndarray]  # H x ~1/eps0 elements in [0,H]
    d: int  # number of values at H

    def __init__(
            self,
            M: MDP,
            eps0: float
        ):
        """
        Constructor method for class DiscretizedMDP.

        Input arguments:
        - M: the tabular MDP to discretize
        - eps0: the discretization parameter (covering radius)
        """
        if M.H < 2 or np.max(M.r) > 1:
            raise Exception("Bad values for the MDP")
        
        self.M = M
        self.eps0 = eps0
        
        # discretization
        self.y_values = [np.zeros(1)]+[np.arange(0, h+eps0, eps0) for h in range(1, M.H+1)]
        self.d = len(self.y_values[-1])

    def compute_optimal_policy(
            self,
            U: np.ndarray
        ) -> List[np.ndarray]:
        """
        Compute the optimal policy for the RS-MDP with the utility U in input.

        Input arguments:
        - U: the utility to use for optimization

        This method returns a list of np.ndarray representing the optimal
        policy. The list contains H items, while the np.ndarray has size
        SxY.
        """

        # construct Q and pi to H, and V to H+1
        Q = [-1*np.ones((self.M.S, len(self.y_values[h]), self.M.A))
             for h in range(self.M.H)]  # H x (S x Y x A) -> R
        pi = [-1*np.ones((self.M.S, len(self.y_values[h])))
             for h in range(self.M.H)]  # H x (S x Y) -> A
        V = [-1*np.ones((self.M.S, len(self.y_values[h])))
             for h in range(self.M.H+1)]  # H+1 x (S x Y) -> R

        # initialize V at H+1
        for s in range(self.M.S):
            V[-1][s,:] = np.copy(U)

        # backward induction from H-1
        for h in range(self.M.H-1, -1, -1):
            for s in range(self.M.S):
                for i,y in enumerate(self.y_values[h]):
                    for a in range(self.M.A):
                        next_y = int((y+self.M.r[s,a,h])/self.eps0)
                        Q[h][s,i,a] = np.dot(V[h+1][:,next_y].ravel(), self.M.p[s,a,h,:])
                    pi[h][s,i] = np.argmax(Q[h][s,i,:])
                    V[h][s,i] = Q[h][s,i,int(pi[h][s,i])]

        return pi
    
    def estimate_return_distribution(
            self,
            pi: List[np.ndarray],
            n_traj: int
        ) -> np.ndarray:
        """
        Sample some returns to estimate the return distribution of input policy
        pi. The episodes are collected using the true reward function (and not
        the discretized one). For state-cumulative reward s,y pairs not covered
        by the input policy pi, we use its interpolation, as described in the
        paper.

        Input arguments:
        - pi: the policy whose return distribution has to be estimated
        - n_traj: the number of trajectories to use for estimation

        This method returns a np.ndarray representing the estimate of the return
        distribution.
        """
        # initialize distribution
        eta = np.zeros(self.d)

        G = (self.sample_returns(pi=pi,n_traj=n_traj)/self.eps0).astype(int)
        for g in G:
            eta[g] += 1

        # normalize
        eta = eta / n_traj

        return eta

    def sample_returns(
            self,
            pi: List[np.ndarray],
            n_traj: int
        ) -> list:
        """
        Sample n_traj trajectories from the underlying MDP using policy pi.
        For each trajectory, compute its return. Then, return a list containing
        all these n_traj return values.

        Input arguments:
        - pi: the policy to simulate in the underlying MDP
        - n_traj: the number of trajectories to simulate

        Returns a list of n_traj sample returns obtained by playing policy pi
        from s0 in M (using the true reward function).
        """

        s = np.full(n_traj, self.M.s_0)

        y = np.zeros(n_traj)

        for h in range(self.M.H):
            a = (pi[h][s, (y/self.eps0).astype(int)]).astype(int)

            y_prime = y + self.M.r[s,a,h]

            if h != (self.M.H-1):
                s_prime = np.array([
                    np.random.choice(self.M.S, 1, p=self.M.p[state,action,h,:].flatten())
                    for state, action in zip(s, a)
                ]).flatten()

                s = s_prime
                y = y_prime

        return y_prime
    