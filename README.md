# Learning Utilities from Demonstrations in Markov Decision Processes

This repository contains the code and the data needed for running the
experiments of the paper.

## Repository Organization

The repository is organized as follows:
- In folder ```data``` there are two .csv files containing the data provided by
  the participants. These are imported and parsed through the functions and
  methods provided inside the ```data/data.py``` file.
- The algorithms are implemented in file ```algorithm.py```. It provides
  implementation for both *CATY-UL* and *TRACTOR-UL*.
- In ```environment.py``` we implement classes *MDP*, modelling a tabular MDP,
  and class *DiscretizedMDP*, which models a discretization of an MDP.
- File ```utils.py``` provides some utilities for plotting, and a method for
  constructing some baseline utilities.
- The experiments described in the paper can be executed by running file
  ```exp1.ipynb```, ```exp2.py```, ```exp2sim.py``` (exp2sim represents the
  second part of experiment 2), and ```exp_add.py``` (exp_add is the additional
  experiment reported in the appendix). The results of the simulations will be
  saved in folders ```results/exp2```,  ```results/exp2sim```, and
  ```results/exp_add``` (exp1 does not save anything).
- The results of experiment 2 and the additional experiment can be analysed
  through notebooks ```exp2 - analysis.ipynb```, ```exp2sim - analysis.ipynb```,
  and ```exp_add - analysis.ipynb```. In particular, these notebooks will
  save many plots in folders ```plots/plot 1```, ```plots/plot 2```,
  ```plots/plot 3```, ```plots/plot 4```, ```plots/plot 5```, and ```plots/plot
  exp2sim```.
- Finally, notebook ```preliminary plots.ipynb``` permits to create in folder
  ```plots/preliminary plots``` plots about some utility functions.

## Requirements

To execute the code, some Python packages are required. Specifically:
- cvxpy (1.5.2)
- pandas (2.2.2)
- matplotlib (3.9.0)
- numpy (1.26.4)

The version number in parenthesis represents the version that we used for
developing the code.

## Datasets

The two datasets used for the experiments are contained into the ```data```
folder. Both datasets contain 15 entries collected in a completely anonymous
manner from 15 participants. Below, we describe the meaning of the entries of
each dataset.

```data_SG.csv```:

Each participant has asked to 8 Standard Gamble (SG) questions, comparing a sure
amount of money $x$ in [10€, 30€, 50€, 100€, 300€, 500€, 1000€, 2000€] with a
lottery between 5000€ and 0€. On the rows, there are the participants, while on
the columns there are the specific values of $x$. Entry $(i,j)$ (row/participant $i$
and column/money $j$) represents the probability that participant i assigns to the
standard gamble between quantity $x$ (for sure), and winning 5000€ (against
0€).

```data_MDP.csv```:

Again, the participants are on the rows, while on the columns we find all the
stage-state-cumulative reward ($h,s,y$) triples for which we asked the
participants to prescribe an action. Each entry represents the specific action
in $\{a_0,a_+,a_-\}$ prescribed by the participant.