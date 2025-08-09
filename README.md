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

## Acknowledgements

This project was made as the final project for my Graphical Models class at UCI. 

I would like to thank Prof. Alexander Ihler for his guidance, class materials, and the [pyGMs](https://github.com/ihler/pyGMs) library used in this project (see `pyGMs-license.txt`).
