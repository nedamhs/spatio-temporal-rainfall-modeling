import numpy as np
import pyGMs as gm
import pyGMs.ising
from pyGMs import Factor, Var
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.special import logsumexp

def my_fit_chowliu(self, data, penalty=0, weights=None):
    """
    Select a maximum likelihood tree-structured graph & parameters
    data: (n,m) nparray of m data points; values {0,1}

    modified version of fit_chowliu function from pyGMs to get mutual info of edges
    (c) 2020 Alexander Ihler under the FreeBSD license; see pyGMs-license.txt for details.
    https://github.com/ihler/pyGMs/blob/master/pyGMs/ising.py
    """
    def MI2(data, weights):
        """Estimate mutual information between all pairs of *binary* {0,1} variables"""
        pi = np.average(data.astype(float),axis=1,weights=weights)[np.newaxis,:]
        pij = np.cov(data,ddof=0,aweights=weights) + (pi.T.dot(pi));
        p = np.stack((pij, pi-pij, pi.T-pij, 1+pij-pi-pi.T), axis=2)
        p2 = pi.T.dot(pi)
        q = np.stack( (p2,pi-p2,pi.T-p2,1+p2-pi-pi.T), axis=2)
        MI = (p*(np.log(p+1e-10)-np.log(q+1e-10))).sum(axis=2)
        return MI,pij,pi[0]

    n,m = data.shape
    MI, pij,pi = MI2(data, weights)       # data should be 0/1, not -1/+1
    tree = mst(penalty-MI).tocoo();

    factors = [Factor([Var(i,2)], [1-pi[i],pi[i]]) for i in range(n)]

    self.mutual_info_edges = []     # to store MI

    for i,j,w in zip(tree.row,tree.col,tree.data):
        if w>0:
          continue
        (i,j)=(int(i),int(j)) if i<j else (int(j),int(i))

        tij = [1+pij[i,j]-pi[i]-pi[j], pi[i]-pij[i,j], pi[j]-pij[i,j], pij[i,j]]
        fij = Factor([Var(i,2),Var(j,2)],tij);
        fij = fij / fij.sum([i]) / fij.sum([j])
        factors.append(fij)

        self.mutual_info_edges.append((i, j, MI[i, j])) #store mutual info

    self.__init__(factors)


# chow-liu tree emissions
class emitChowLiu:
    def __init__(self, data=None, penalty = 0):
        """Initialize the distribution using data, if provided"""
        self.model = gm.ising.Ising()
        self.penalty = penalty
        if data is not None:
            # initially fit without weights
            my_fit_chowliu(self.model, data.T, penalty=self.penalty)

    def logLikelihood(self, data):
        """Evaluate the log-probability log p_z(x) for this distribution"""
        """Use pseudo-log-likelihood as proxy for log p(x | z)"""
        pll = self.model.pseudolikelihood(data)
        return pll + 1e-8

    def fit(self, data, weights):
        """Fit weighted Chow-Liu model"""
        # fit chow liu w weights
        my_fit_chowliu(self.model, data.T, penalty=self.penalty, weights=weights.flatten())


# Baum-Welch Algorithm
# code borrowed from https://sli.ics.uci.edu/extras/cs179/Demo%20-%20ActivityHMM.html
class HMM(object):
    def __init__(self, T, emitters, p0=None):
        """Initialize an HMM.  T=p(Zt|Zt-1), emitters = [p(X|Z=0) ... ]"""
        nx = len(emitters)
        self.T = T
        self.T /= self.T.sum(0, keepdims=True)
        self.E = emitters
        self.P0 = p0 if p0 is not None else np.ones(self.T.shape[0]) / self.T.shape[0]   #initial probs
        self.T_valid = 1  # (n,n) array, =1 for valid transitions, 0 for forbidden
        self.T_prior = 0  # (n,n) array, pseudo-counts for transitions

    def forward_backward(self, x):
        """Perform forward-backward inference to compute p(Z)'s marginals
           Returns log p(x), forward msgs, backward msgs, p(Zt), and p(xt|Zt)"""
        L = len(x)
        n = self.T.shape[0]
        alph, beta, obs, prob = (np.zeros((L, n)) for i in range(4))

        for i in range(n):
            obs[:,i] = np.exp(self.E[i].logLikelihood(x))
            # print(self.E[i].logLikelihood(x))

        # for i in range(n):
        #       obs[:, i] = self.E[i].logLikelihood(x)
        # obs = np.exp(obs - logsumexp(obs, axis=1, keepdims=True))

        alph[0, :] = self.P0 * obs[0, :]
        zt = alph[0, :].sum()
        logl = np.log(zt)
        alph[0, :] /= zt

        for l in range(1, L):
            alph[l, :] = alph[l - 1, :].dot(self.T) * obs[l, :]
            zt = alph[l, :].sum()
            logl += np.log(zt)
            alph[l, :] /= zt

        beta[L - 1, :] = 1.
        prob[L - 1, :] = alph[L - 1, :]
        for l in range(L - 2, -1, -1):
            beta[l, :] = (beta[l + 1, :] * obs[l + 1, :]).dot(self.T.T)
            beta[l, :] /= beta[l, :].sum()
            prob[l, :] = alph[l, :] * beta[l, :]
            prob[l, :] /= prob[l, :].sum()

        return logl, alph, beta, prob, obs

    def EM(self, x):
        """Perform EM to train the HMM. Returns log p(x)'s value *before* the update"""
        n, L = self.T.shape[0], len(x)

        # "EXPECTATION" -- compute required probability values
        logl, alph, beta, prob, obs = self.forward_backward(x)
        # print(logl)

        # "MAXIMIZATION" -- compute ML estimates of parameters: T, E
        tmp = 0 * self.T
        for i in range(L - 1):
            P = alph[i, :].reshape(-1, 1) * self.T * (obs[i + 1, :] * beta[i + 1, :]).reshape(1, -1)
            tmp += P / P.sum()
        tmp += self.T_prior
        tmp *= self.T_valid
        self.T = tmp / tmp.sum(1, keepdims=True)

        for i in range(n):
            self.E[i].fit(x, prob[:, i])

        return logl