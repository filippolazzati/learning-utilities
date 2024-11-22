from typing import Tuple
import numpy as np
import pandas as pd
from environment import *

def construct_env_utilities_and_policies() -> Tuple[DiscretizedMDP, List[np.ndarray], List[list]]:
            """
            This method constructs 3 objects:
            1) a DiscretizedMDP, which is the environment described in the
            paper;
            2) a list of utilities, which are fitted using the data collected
            through the Standard Gamble (SG) in the form;
            3) a list of policies, representing the policies of the participants
            to the data collection phase.

            It returns a tuple containing these objects.
            """
            M = get_env()
            utilities = get_utilities(M)
            policies = get_policies(M)

            return (M, utilities, policies)

def get_env() -> DiscretizedMDP:
        """
        Construct the instance of DiscretizedMDP described in the paper.
        """
        # S = {L, M, H, T} <---> {0, 1, 2, 3}
        S = 4
        # A = {a0, a+, a-} <---> {0, 1, 2}
        A = 3
        # H (number of actions to take)
        H = 5
        # s_0 = M
        s_0 = 1
        # epsilon_0
        eps_0 = 0.01
        # alpha +
        a = 1/3*np.ones(H)
        # alpha -
        b = 1/5*np.ones(H)
        # k +
        c = 0
        # k -
        d = 2

        # r = S x A x H
        r_L = 0*np.ones(H)
        r_M = 30*np.ones(H)
        r_H = 100*np.ones(H)
        r_T = 500*np.ones(H)
        r = np.array([[r_L, c*r_L, d*r_L],
                    [r_M, c*r_M, d*r_M],
                    [r_H, c*r_H, d*r_H],
                    [r_T, c*r_T, d*r_T]])
        # normalize r to [0,1]
        r /= np.max(r)

        # p = S x A x H x S
        ones = np.ones(H)
        zeros = np.zeros(H)
        # initialize as S x A x S x H
        p = np.array([[[ones, zeros, zeros, zeros],
                    [ones-a, a, zeros, zeros],
                    [ones, zeros, zeros, zeros]],
                    [[zeros, ones, zeros, zeros],
                    [zeros, ones-a, a, zeros],
                    [b, ones-b, zeros, zeros]],
                    [[zeros, zeros, ones, zeros],
                    [zeros, zeros, ones-a, a],
                    [zeros, b, ones-b, zeros]],
                    [[zeros, zeros, zeros, ones],
                    [zeros, zeros, zeros, ones],
                    [zeros, zeros, b, ones-b]]])
        # convert to S x A x H x S
        p = np.transpose(p, (0, 1, 3, 2))

        # MDP
        M1 = MDP(S=S, A=A, H=H, r=r, p=p, s_0=s_0)

        # DiscretizedMDP
        M = DiscretizedMDP(M=M1,eps0=eps_0)

        return M

def get_utilities(M: DiscretizedMDP) -> List[np.ndarray]:
        """
        Read the data from file, and use it to construct a list containing the
        utility functions obtained through SG of the participants to the data
        collection phase.        
        """
        # read the data
        df = pd.read_csv('data/data_SG.csv')
        data = df.to_numpy()

        utilities = []       

        # construct the utilities
        for row in data:
            x = np.array([0,1,3,5,10,30,50,100,200,500])
            U = np.array([0,row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],1])

            # normalize in [0,H]
            U *= M.M.H

            # fill empty values with linear interpolation
            xx = np.linspace(0,500,num=M.d)
            UU = np.interp(xx, x, U)
            
            utilities.append(UU)

        return utilities

def get_policies(M: DiscretizedMDP) -> List[list]:
        """
        Read the data from file, and construct the policies representing the
        strategies of the participants in the MDP. Use some kind of
        interpolation to fill the policies in other values of cumulative reward.
        """
        # read the data
        df = pd.read_csv('data/data_MDP.csv')
        data = df.to_numpy()

        # convert action labels to numbers
        data = np.where(data == 'a0', 0, data)
        data = np.where(data == 'a+', 1, data)
        data = np.where(data == 'a-', 2, data)

        policies = []

        # construct the policies
        for row in data:
            # use value -1 to indicate that the action is not needed
            piE = [-np.ones((M.M.S, len(M.y_values[h]))) for h in range(M.M.H)]

            # from state L, since all actions give 0$, only action a+ is rational
            piE[1][0,:] = 1
            piE[2][0,:] = 1
            piE[3][0,:] = 1
            piE[4][0,:] = 1

            # at H=5 always play a-
            piE[4][:,:] = 2

            ### h=1:

            # at 1, play only from s=M, y=0
            piE[0][1,0] = row[0]

            ### h=2:

            # at 2, play only from M and H. With eps0=0.01, every cell
            # represents 10$
            # M:
            piE[1][1,0] = row[1]
            piE[1][1,3] = row[2]
            piE[1][1,6] = row[3]
            # H:
            piE[1][2,0] = row[4]

            ### h=3:

            # at 3, play from M, H and T.
            # M:
            piE[2][1,0] = row[5]
            piE[2][1,3] = row[6]
            piE[2][1,6] = row[7]
            piE[2][1,20] = row[8]
            # H:
            piE[2][2,0] = row[9]
            piE[2][2,3] = row[10]
            piE[2][2,6] = row[11]
            piE[2][2,20] = row[12]
            # T:
            piE[2][3,0] = row[13]

            ### h=4:

            # at 4, play from M, H and T.
            # M:
            piE[3][1,0] = row[14]
            piE[3][1,3] = row[15]
            piE[3][1,6] = row[16]
            piE[3][1,9] = row[17]
            piE[3][1,12] = row[18]
            piE[3][1,15] = row[19]
            piE[3][1,18] = row[20]
            piE[3][1,30] = row[21]
            piE[3][1,40] = row[22]
            # H:
            piE[3][2,0] = row[23]
            piE[3][2,3] = row[24]
            piE[3][2,6] = row[25]
            piE[3][2,10] = row[26]
            piE[3][2,13] = row[27]
            piE[3][2,20] = row[28]
            piE[3][2,30] = row[29]
            piE[3][2,100] = row[30]
            # T:
            piE[3][3,0] = row[31]
            piE[3][3,6] = row[32]
            piE[3][3,10] = row[33]
            piE[3][3,20] = row[34]
            piE[3][3,50] = row[35]
            piE[3][3,100] = row[36]

            ### fill empty values with surrounding values where needed

            # for stages h=3 and h=4, we have to fill the -1 with
            # the action played with cumulative reward closest to it

            # h=3, M:
            piE[2][1,1] = piE[2][1,0]
            piE[2][1,2] = piE[2][1,3]
            piE[2][1,4] = piE[2][1,3]
            piE[2][1,5] = piE[2][1,6]
            piE[2][1,7:14] = piE[2][1,6]
            piE[2][1,14:20] = piE[2][1,20]
            piE[2][1,21:] = piE[2][1,20]

            # h=3, H
            piE[2][2,1] = piE[2][2,0]
            piE[2][2,2] = piE[2][2,3]
            piE[2][2,4] = piE[2][2,3]
            piE[2][2,5] = piE[2][2,6]
            piE[2][2,7:14] = piE[2][2,6]
            piE[2][2,14:20] = piE[2][2,20]
            piE[2][2,21:] = piE[2][2,20]

            # h=4, M:
            piE[3][1,1] = piE[3][1,0]
            piE[3][1,2] = piE[3][1,3]
            piE[3][1,4] = piE[3][1,3]
            piE[3][1,5] = piE[3][1,6]
            piE[3][1,7] = piE[3][1,6]
            piE[3][1,8] = piE[3][1,9]
            piE[3][1,10] = piE[3][1,9]
            piE[3][1,11] = piE[3][1,12]
            piE[3][1,13] = piE[3][1,12]
            piE[3][1,14] = piE[3][1,15]
            piE[3][1,16] = piE[3][1,15]
            piE[3][1,17] = piE[3][1,18]
            piE[3][1,19:25] = piE[3][1,18]
            piE[3][1,25:30] = piE[3][1,30]
            piE[3][1,31:36] = piE[3][1,30]
            piE[3][1,36:40] = piE[3][1,40]
            piE[3][1,41:] = piE[3][1,40]

            # h=4, H:
            piE[3][2,1] = piE[3][2,0]
            piE[3][2,2] = piE[3][2,3]
            piE[3][2,4] = piE[3][2,3]
            piE[3][2,5] = piE[3][2,6]
            piE[3][2,7:9] = piE[3][2,6]
            piE[3][2,9] = piE[3][2,10]
            piE[3][2,11] = piE[3][2,10]
            piE[3][2,12] = piE[3][2,13]
            piE[3][2,14:17] = piE[3][2,13]
            piE[3][2,17:20] = piE[3][2,20]
            piE[3][2,21:26] = piE[3][2,20]
            piE[3][2,26:30] = piE[3][2,30]
            piE[3][2,31:66] = piE[3][2,30]
            piE[3][2,66:100] = piE[3][2,100]
            piE[3][2,101:] = piE[3][2,100]

            # h=4, T:
            piE[3][3,1:4] = piE[3][3,0]
            piE[3][3,4:6] = piE[3][3,6]
            piE[3][3,7:9] = piE[3][3,6]
            piE[3][3,9] = piE[3][3,10]
            piE[3][3,11:16] = piE[3][3,10]
            piE[3][3,16:20] = piE[3][3,20]
            piE[3][3,21:36] = piE[3][3,20]
            piE[3][3,36:50] = piE[3][3,50]
            piE[3][3,51:76] = piE[3][3,50]
            piE[3][3,76:100] = piE[3][3,100]
            piE[3][3,101:] = piE[3][3,100]

            ### append

            policies.append(piE)

        return policies

def get_env_piE_random(
              S: int,
              A: int,
              H: int,
              eps_0: float,
) -> Tuple[DiscretizedMDP, list]:
        """
        Construct a random instance of DiscretizedMDP with size of the state and
        action spaces respectively S and A. S must be >= 2. Moreover, randomly
        generate a deterministic expert's policy in this env.
        """
        # s_0 = M
        s_0 = 1

        # r = S x A x H, r in [0,1]
        r = np.random.rand(S, A, H)

        # p = S x A x H x S
        p = np.zeros((S, A, H, S))
        for s in range(S):
                for a in range(A):
                        for h in range(H):
                                random_values = np.random.rand(S)
                                p[s,a,h,:] = random_values / random_values.sum()

        # MDP
        M1 = MDP(S=S, A=A, H=H, r=r, p=p, s_0=s_0)

        # DiscretizedMDP
        M = DiscretizedMDP(M=M1,eps0=eps_0)

        # piE = H x (S x Y) -> A
        piE = [np.zeros((S, len(M.y_values[h]))) for h in range(H)]
        for h in range(H):
                piE[h][:,:] = np.random.randint(0, A, (S, len(M.y_values[h])))  # A excluded

        return (M, piE)

def construct_random_envs_and_policies(
                N: int,
                S: int,
                A: int,
                H: int,
                eps_0: float,
                seed: int
) -> Tuple[List[DiscretizedMDP], List[list]]:
            """
            Construct N pairs (MDP, piE) with state and action spaces with
            cardinalities S and A. The initial state will be s_0=1, and the
            discretization eps_0=0.01.
            """
            np.random.seed(seed)

            Ms = []
            piEs = []

            for _ in range(N):
                    M, piE = get_env_piE_random(S=S, A=A, H=H, eps_0=eps_0)
                    
                    Ms.append(M)
                    piEs.append(piE)

            return Ms, piEs
