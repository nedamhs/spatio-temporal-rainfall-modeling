
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from src.hmm import * 

extent = (65.,95.,5.,36.)

coastx = [[81.79,81.64,81.22,80.35,79.87,79.7,80.15,80.84,81.3,81.79],
      [95.38,95.29,95.94],
      [95.37,94.81,94.19,94.53,94.32,93.54,93.66,93.08,92.37,92.08,92.03,91.83,91.42,90.5,90.59,90.27,89.85,89.7,89.42,89.03,88.89,88.21,86.98,87.03,86.5,85.06,83.94,83.19,82.19,82.19,81.69,80.79,80.32,80.03,80.23,80.29,79.86,79.86,79.34,78.89,79.19,78.28,77.94,77.54,76.59,76.13,75.75,75.4,74.86,74.62,74.44,73.53,73.12,72.82,72.82,72.63,71.18,70.47,69.16,69.64,69.35,68.18,67.44,67.15,66.37,64.53]]
      
coasty = [[7.52,6.48,6.2,5.97,6.76,8.2,9.82,9.27,8.56,7.52],
      [4.97,5.48,5.44],
      [15.71,15.8,16.04,17.28,18.21,19.37,19.73,19.86,20.67,21.19,21.7,22.18,22.77,22.81,22.39,21.84,22.04,21.86,21.97,22.06,21.69,21.7,21.5,20.74,20.15,19.48,18.3,17.67,17.02,16.56,16.31,15.95,15.9,15.14,13.84,13.01,12.06,10.36,10.31,9.55,9.22,8.93,8.25,7.97,8.9,10.3,11.31,11.78,12.74,13.99,14.62,15.99,17.93,19.21,20.42,21.36,20.76,20.88,22.09,22.45,22.84,23.69,23.94,24.66,25.43,25.24]]

def draw_map():
    """
    Plot coastline segments using coastx/coasty within the given extent.
    code borrowed from https://sli.ics.uci.edu/extras/cs179/old/Demo%20-%20Ising%20Rainfall.html
    """
    for x,y in zip(coastx,coasty): 
         plt.plot(x,y,'k-',color=(.4,.4,.4));
    plt.axis(extent)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('rainfall stations across India')


def draw_chowLiu_tree(ising_cl, lat, lon, vmin, vmax, rain_vals=None, ax=None, global_min_mi=None, global_max_mi=None):
    """
    Draw a Chow-Liu tree on a geographic map.
    Nodes are colored by avg rainfall frequency of the station, and edge widths represent pairwise mutual information between stations.
    Node colors and edge widths are normalized. 
    """
    G = nx.Graph()
    edge_weights = {}
    mi_edges = getattr(ising_cl, "mutual_info_edges", None)  # mutual info for edges
    # print(mi_edges)

    # build graph from pairwise factors
    for f in ising_cl.factors:
        if len(f.vars) == 2:
            i, j = map(int, f.vars)
            G.add_edge(i, j)
            if mi_edges:
                for a, b, mi in mi_edges:
                    if {a, b} == {i, j}:
                        edge_weights[(i, j)] = mi
                        break
    pos = {i: (lon[i], lat[i]) for i in G.nodes}

    # Edge widths based on Mutual Info (normalized)
    if edge_weights:
        if global_min_mi is None or global_max_mi is None:
            all_mis = np.array(list(edge_weights.values()))
            min_mi, max_mi = all_mis.min(), all_mis.max()
        else:
            min_mi, max_mi = global_min_mi, global_max_mi
        widths = [1 + 6 * (edge_weights.get((i, j), 0) - min_mi) / (max_mi - min_mi + 1e-6) for i, j in G.edges] #edge width

    # Node color based on rainfall frequency values (normalized)
    if rain_vals is not None:
        rain_vals = np.asarray(rain_vals)
        node_colors = rain_vals[list(G.nodes)]
        # vmin, vmax = rain_vals.min(), rain_vals.max()
        vmin, vmax = 0, vmax      # vmax = globale max rainfall freq 

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color=node_colors, cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=widths)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)


def compute_bic(logl, n, d, N):
    """Compute BIC for HMM with n states, d-dimensional emissions, and N data points."""
    # free parameters:
    k = (n * (n - 1)      # off-diagonal transition probs
        + (n - 1)         # initial state probs
        + n * (2 * d - 1) # emission parameters
    )
    bic = -2 * logl + k * np.log(N)
    return bic


def LOOCV(labels, n_states):
    """
    Perform 10-fold leave-one-year-out cross-validation for HMM.

    Inputs:
        labels   : Binary observation matrix of shape (N, d).
        n_states : Number of hidden states in the HMM.

    Returns:
        total_ll    : Total predictive log-likelihood across all folds.
        pll_per_day : Average predictive log-likelihood per day.
    """
    N = len(labels)     # 3652 days
    d = labels.shape[1] # 53 stations

    test_lls = []
    for fold in range(10):
        # train / test indices
        test_start = fold * 365
        test_end = test_start + 365
        test_idx = np.arange(test_start, test_end)
        train_idx = np.setdiff1d(np.arange(N), test_idx)

        x_train = labels[train_idx]    # train data
        x_test = labels[test_idx]      # test data

        emitters = [emitChowLiu(x_train[np.random.choice(len(x_train), size=1000, replace=False)]) for _ in range(n_states)]
        model = HMM(np.eye(n_states) * 0.9 + 0.1, emitters)
        model.T_prior = np.eye(n_states) * 50000.

        for i in range(50):  
            model.EM(x_train)   #train model 

        # eval log-likelihood on the test set ( Actually PLL Here )
        logl, *_ = model.forward_backward(x_test)
        test_lls.append(logl)
        #print(f"leaving out Year {fold + 1}/{10}, test ll: {logl:.2f}")

    total_ll = np.sum(test_lls)
    pll_per_day = total_ll / N

    return total_ll, pll_per_day