from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_utilities(
        U: List[np.ndarray],
        labels: List[str],
        ylim: bool = False,
        title: str = '',
        normal_legend_size: bool = False,
        savefig: bool = False,
        namefig: str = ''
        ):
        """
        Take a list of utilities and plot them.

        Input arguments:
        - U: the list of utilities to plot
        - labels: a list of labels for each utility
        - ylim: whether to limit the y values or not
        - title: optional, the title of the plot
        - normal_legend_size: optional, whether to use a normal legend size
        - savefig: whether to save the figure to file
        - namefig: the name to assign to the saved figure
        """
        plt.figure(figsize=(10, 6))

        # create a colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(U)))

        # plot the utilities
        for i, ut in enumerate(U):
            plt.plot(ut, linestyle='-', linewidth=3, color=colors[i], label=labels[i])

        # adding title and labels
        plt.xlabel('Return $G$', fontsize=20)
        plt.ylabel('Utility $U(G)$', fontsize=20)
        if not normal_legend_size:
            plt.legend(fontsize=18)
        else:
            plt.legend()
        plt.grid(True)
        plt.xticks(ticks=[0,100,200,300,400,500], labels=[0,1,2,3,4,5], fontsize=16)
        plt.yticks(fontsize=16)
        if ylim:
            plt.ylim(top=5.3, bottom=-0.3)
        if title != '':
            plt.title(title)

        # save the image
        if savefig:
            plt.savefig(namefig+".pdf", format="pdf", dpi=1200)

        # show the plot
        plt.show()


def plot_suboptimalities(
        subopts: List[np.ndarray],
        subopts_std: List[np.ndarray] = [],
        labels: List[str] = [],
        title: str = '',
        savefig: bool = False,
        namefig: str = ''
):
    """
    Take a list of suboptimalities, i.e., (non) compatibilities,
    representing the error per iteration of some algorithms, and
    plot it.
    
    Input arguments:
    - subopts: the list of suboptimalities to plot
    - subopts_std: optional, takes a list of stds of the subopts to plot
    - labels: optional, a list of labels for each suboptimality
    - title: optional, the title of the plot
    - savefig: whether to save the figure to file
    - namefig: the name to assign to the saved figure
    """
    plt.figure(figsize=(10, 6))

    # create a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(subopts)))

    # plot the suboptimalities
    if len(subopts) > 1:
        for i, sub in enumerate(subopts):
            plt.plot(sub, linestyle='-', linewidth=3, color=colors[i], label=labels[i])
            # fill the area using the stds
            plt.fill_between(range(len(sub)), sub - subopts_std[i], sub + subopts_std[i], color=colors[i], alpha=0.3)
    else:
        plt.plot(subopts[0], linestyle='-', linewidth=3)

    # adding title and labels
    plt.xlabel('Iteration $t$', fontsize=20)
    plt.ylabel('(Non)compatibility '+r'$\overline{\mathcal{C}}(\widehat{U}_t)$', fontsize=20)
    if len(subopts) > 1:
        plt.legend(fontsize=18)    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    if title != '':
        plt.title(title)

    # save the image
    if savefig:
        plt.savefig(namefig+".pdf", format="pdf", dpi=1200)

    # show the plot
    plt.show()
    

def plot_return_distributions(
        etas: List[np.ndarray],
        labels: List[str],
        title: str = '',
        huge_text: bool = False,
        savefig: bool = False,
        namefig: str = ''
):
    """
    Take a list of return distributions and plot them.

    Input arguments:
    - etas: the list of return distributions to plot
    - labels: a list of labels for each return distribution
    - title: optional, the title of the plot
    - huge_text: optional, to increase the font size
    - savefig: whether to save the figure to file
    - namefig: the name to assign to the saved figure
    """
    plt.figure(figsize=(10, 6))

    # create a colormap
    colors = plt.cm.Set1(np.linspace(0, 1, len(etas)))

    # plot the etas
    for i, eta in enumerate(etas):
        plt.plot(eta, marker='.', markersize=20, color=colors[i], label=labels[i])

    # adding title and labels
    if huge_text:
        plt.xlabel('Return $G$', fontsize=32)
        plt.ylabel('Probability $\eta(G)$', fontsize=32)
        plt.legend(fontsize=32)
        plt.xticks(ticks=[0,100,200,300,400,500], labels=[0,1,2,3,4,5], fontsize=25)
        plt.yticks(fontsize=25)
    else:
        plt.xlabel('Return $G$', fontsize=20)
        plt.ylabel('Probability $\eta(G)$', fontsize=20)
        plt.legend(fontsize=18)
        plt.xticks(ticks=[0,100,200,300,400,500], labels=[0,1,2,3,4,5], fontsize=16)
        plt.yticks(fontsize=16)
    plt.grid(True)
    if title != '':
        plt.title(title)

    # save the image
    if savefig:
        if huge_text:
             plt.savefig(namefig+".pdf", format="pdf", dpi=1200, bbox_inches='tight')
        else:
            plt.savefig(namefig+".pdf", format="pdf", dpi=1200)

    # show the plot
    plt.show()


def plot_difference_return_distributions(
        delta: np.ndarray,
        difference_name: str,
        title: str = '',
        savefig: bool = False,
        namefig: str = ''
):
    """
    Take a list of return distributions and plot them.

    Input arguments:
    - delta: the delta of return distributions to plot
    - difference_name: a string with the names of the etas
    - title: optional, the title of the plot
    - savefig: whether to save the figure to file
    - namefig: the name to assign to the saved figure
    """
    plt.figure(figsize=(10, 6))

    # plot the delta
    plt.plot(delta, marker='.', markersize=20, color='b')

    # adding title and labels
    plt.xlabel('Return $G$', fontsize=20)
    plt.ylabel('Difference $'+difference_name+'$', fontsize=20)
    plt.grid(True)
    plt.xticks(ticks=[0,100,200,300,400,500], labels=[0,1,2,3,4,5], fontsize=16)
    plt.yticks(fontsize=16)
    if title != '':
        plt.title(title)

    # save the image
    if savefig:
        plt.savefig(namefig+".pdf", format="pdf", dpi=1200)

    # show the plot
    plt.show()


def get_utility(
        function: str,
        H: int,
        d: int
    ) -> np.ndarray:
    """
    Construct a utility function based on the name passed
    as argument. It can be used as initial utility for algorithm
    TRACTOR-UL.

    Input arguments:
    - function: the name of the utility to construct. Possible values
    are ['sqrt', 'square', 'linear']
    - H: the maximum input value to the utility function. It coincides with
    the horizon H of some MDP
    - d: the discretization of the utility. Coincides with the discretization
    of the plausible return vectors in a DiscretizedMDP

    It returns a np.ndarray representing the constructed utility.
    """
    # initialize grid of points
    x = np.linspace(0, H, d)

    # construct vectorized function
    if function == 'sqrt':
        fun = np.vectorize(lambda x: np.sqrt(x))
    elif function == 'square':
        fun = np.vectorize(lambda x: x**2)
    elif function == 'linear':
        fun = np.vectorize(lambda x: x)
    else:
        raise Exception('Utility name unknown!')
    
    # create utility
    U = fun(x)

    # normalize utility in [0,H]
    U /= U[-1]
    U *= H

    return U
