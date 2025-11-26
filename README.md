# spatio temporal rainfall modeling

 Hidden Markov Models (HMMs) with Chow–Liu tree emissions for modeling seasonal rainfall patterns in India


This project is inspired by [this paper](https://arxiv.org/abs/1207.4142):

> Kirshner, S., Smyth, P., & Robertson, A. W. (2004). Conditional Chow–Liu tree structures for modeling discrete-valued vector time series. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI'04)*, pp. 317–324. AUAI Press.


## Hidden Markov Models

**Latent Temporal states:**

The HMM models daily rainfall occurrence as a sequence of hidden states, with each state representing a distinct seasonal rainfall pattern.

 <img src="assets/state_0.gif" width="265" />  <img src="assets/state_1.gif" width="265" />  <img src="assets/state_2.gif" width="265" /> 

 <img src="assets/states.png" width="950" />
 

## HMM with Chow–Liu Tree Emissions 

**Learned spatial dependencies:**

The emission distribution p(x|z=k) for each latent state z is modeled using a Chow-Liu tree, which captures spatial dependencies across
stations via a tree-structured Ising model. The edges of the tree are selected to form a maximum-weight spanning tree based on 
mutual information between pairs of stations.

<img src="assets/HMM+CL.png" width="930" height="500" />


* Node colors represent the average rainfall frequency at each station across all days assigned to the corresponding latent state, edge width indicates the strength of Mutual Information between stations.
*  Z = 0 corresponds to transitional season, Z = 1 corresponds to Rainy/Monsoon season, Z=2 corresponds to dry season.



For more details, see [Report.pdf](Report.pdf).

---
## Adding BIC penalty to Chow-Liu Trees
Adding a BIC penalty lets the Chow–Liu procedure prune weak edges, so instead of forcing a fully connected tree, it can return a smaller forest that keeps only the strongest spatial dependencies.

###### add img here 

After adding the BIC penalty, the monsoon state still showed strong coastal links, the transitional state preserved only its major rainy-region edges, and the dry state became almost empty, 
highlighting the much weaker mutual information during the dry season.

---
## Baseline


---
## Model Selection

We use two model selection methods to choose the number of latent temporal states for the HMM.

- **LOOCV:** The data is split into ten folds (years). For each fold, the model is trained on nine years and evaluated on the held-out year, and the held-out pseudo-log-likelihood per day (PLL/day) is used as the model selection metric.
- **BIC:**  Models with different latent state counts are evaluated using the BIC score, which balances fit and complexity, lower BIC score indicate a better trade-off.

  ###### add table here 

---

## Acknowledgements

This project was made as the final project for my Graphical Models class at UCI. 

I would like to thank Prof. Alexander Ihler for his guidance, class materials, and his [pyGMs](https://github.com/ihler/pyGMs) library used in this project.


---

## Code Resources

- The Baum–Welch learning algorithm was borrowed from the CS179 class material ([ActivityHMM](https://ics.uci.edu/~ihler/classes/cs179/Demo%20-%20ActivityHMM.html)).

- The Chow–Liu implementation was borrowed and lightly modified from the [pyGMs](https://github.com/ihler/pyGMs) library (see `pyGMs-license.txt`).

---

## Citations
[1] Kirshner, S., Smyth, P., & Robertson, A. W. (2004). Conditional Chow–Liu tree structures for modeling discrete-valued vector time series.
UAI’04.
https://arxiv.org/abs/1207.4142

[2] Ihler, A. T., et al. Graphical Models for Statistical Inference and Data Assimilation.
https://www.ics.uci.edu/~ihler/papers/physd07.pdf

[3] pyGMs Library (Alexander Ihler).
https://github.com/ihler/pyGMs
