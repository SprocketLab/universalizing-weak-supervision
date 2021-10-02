import numpy as np
import random
from scipy.stats import multivariate_normal

class LabelModel():
    def __init__(self, use_triplets=True):
        self.use_triplets = use_triplets  # only choice right now

    def fit(self, L_train, var_Y, median=False, seed=10):
        self.n, self.m = L_train.shape
        n, m = self.n, self.m
        self.O = np.transpose(L_train) @ L_train / self.n
        self.Sigma_hat = np.zeros([m + 1, m + 1])
        self.Sigma_hat[:m, :m] = self.O

        random.seed(seed)

        if median:
            # Init dict to collect accuracies in triplets
            acc_collection = {}
            for i in range(m):
                acc_collection[i] = []

            # Collect triplet results
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        acc_i = np.sqrt(self.O[i, j] * self.O[i, k] * var_Y / self.O[j, k])
                        acc_j = np.sqrt(self.O[j, i] * self.O[j, k] * var_Y / self.O[i, k])
                        acc_k = np.sqrt(self.O[k, i] * self.O[k, j] * var_Y / self.O[i, j])
                        acc_collection[i].append(acc_i)
                        acc_collection[j].append(acc_j)
                        acc_collection[k].append(acc_k)

            # Take medians
            for i in range(m):
                self.Sigma_hat[i, m] = np.median(acc_collection[i])
                self.Sigma_hat[m, i] = np.median(acc_collection[i])
        else:
            for i in range(m):
                idxes = set(range(m))
                idxes.remove(i)
                # triplet is now i,j,k
                [j, k] = random.sample(idxes, 2)
                # solve from triplet using conditional independence
                acc = np.sqrt(self.O[i, j] * self.O[i, k] * var_Y / self.O[j, k])
                self.Sigma_hat[i, m] = acc
                self.Sigma_hat[m, i] = acc

        # we filled in all but the right-bottom corner, add it in
        self.Sigma_hat[m, m] = var_Y
        return

    def inference(self, L):
        n, m = self.n, self.m
        self.Y_hat = np.zeros(self.n)
        for i in range(self.n):
            self.Y_hat[i] = np.expand_dims(self.Sigma_hat[m, :m], axis=0) \
                            @ np.linalg.inv(self.Sigma_hat[:m, :m]) \
                            @ np.expand_dims(L[i, :self.m], axis=1)
        return

    def score(self, Y_samples, metric="mse"):
        err = 0
        for i in range(self.n):
            err += (Y_samples[i] - self.Y_hat[i]) ** 2
        return err / self.n

def generate_synthetic_xy(n=1000, d=10):
    '''Generate n (X,Y) points, with X in R^d'''
    # X generation:
    X_mu, X_sigma = np.zeros(d), np.diag(np.random.rand(d))
    X_var = multivariate_normal(mean=X_mu, cov=X_sigma)
    X_samples = X_var.rvs(n)

    # Y generation:
    beta = np.ones(d)
    Y_samples = X_samples @ beta

    # Y variance:
    Y_var = np.inner(beta, np.diag(X_sigma))

    return X_samples, Y_samples, Y_var


def generate_lfs(m, n, Y, Y_var, seed=42):
    '''Generate m labeling functions for Y '''

    # K is the inverse covariance matrix for the LFs (precision matrix)
    diag_fact, K_fact = 5.0, -3.0  # Some constants to build K
    K = np.zeros([m + 1, m + 1])
    K[:, m], K[m, :] = np.ones([m + 1]) * K_fact, np.ones([m + 1]) * K_fact
    K[m, m] = 0.0
    K += np.eye(m + 1) * diag_fact  # + np.random.rand(m+1) * 0.1

    # the one trick is that we need Y to be the same as Y_var for the coupling
    # just modify by alpha*e_me_m^T, compute it with Sherman Morrison
    Sigma_temp = np.linalg.inv(K)
    Y_var_temp = Sigma_temp[m, m]
    un = np.zeros(m + 1)
    un[m] = 1.0
    unv = np.expand_dims(un, axis=0)
    ent = Sigma_temp @ np.outer(un, un) @ Sigma_temp
    alpha = (Y_var_temp - Y_var) / (ent[m, m] + (Y_var - Y_var_temp) * unv @ Sigma_temp @ np.transpose(unv))
    Sigma = np.linalg.inv(K + alpha * np.outer(un, un))

    # Generate the LFs.
    # NB: conditional distribution LF | Y ~ N(mu, Sigma)
    Sig_bar = Sigma[:m, :m] - 1.0 / Sigma[m, m] * np.outer(Sigma[m, :m], Sigma[m, :m])
    Lambda_var = multivariate_normal(mean=np.zeros(m), cov=Sig_bar, seed=seed)
    L = Lambda_var.rvs(n)
    means = 1.0 / Sigma[m, m] * np.expand_dims(Sigma[:m, m], axis=1) @ np.expand_dims(Y, axis=0)
    L += np.transpose(means)

    return L, Sigma