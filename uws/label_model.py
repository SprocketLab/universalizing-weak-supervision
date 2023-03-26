import numpy as np
import random
import torch
import geoopt

from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.lorentz import math
from flyingsquid.label_model import LabelModel
from scipy.stats import multivariate_normal

# TODOs
# Add ranking, parsing tree label model
# Ranking: need refactoring due to dependency on ranking object
# Parsing tree

# BinaryLabelModel just use FlyingSquid: https://github.com/HazyResearch/flyingsquid, https://arxiv.org/abs/2002.11955
class BinaryLabelModel(LabelModel):
    def __init__(self, m):
        super().__init__(m)


class ContinuousLabelModel():
    def __init__(self, use_triplets=True):
        self.use_triplets = use_triplets  # only choice right now

    def fit(self, L_train, var_Y, median=True, seed=10):
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

    def predict(self, L):
        n, m = self.n, self.m
        self.Y_hat = np.zeros(self.n)
        for i in range(self.n):
            self.Y_hat[i] = np.expand_dims(self.Sigma_hat[m, :m], axis=0) \
                            @ np.linalg.inv(self.Sigma_hat[:m, :m]) \
                            @ np.expand_dims(L[i, :self.m], axis=1)
        return self.Y_hat

    def score(self, Y_samples, metric="mse"):
        err = 0
        for i in range(self.n):
            err += (Y_samples[i] - self.Y_hat[i]) ** 2
        return err / self.n
    
    
class GeodesicLabelModel():
    def __init__(self, num_triplets=1):
        self.num_triplets = num_triplets
        self.num_lfs = self.num_triplets * 3
        self.thetas = []

    def fit(self, lambdas):
        dim = lambdas[0].shape[1]
        for i in range(self.num_triplets):
            Eab = torch.mean(
                man.dist(lambdas[(i*3)+0], lambdas[(i*3)+1])) / dim
            Ebc = torch.mean(
                man.dist(lambdas[(i*3)+1], lambdas[(i*3)+2])) / dim
            Eac = torch.mean(
                man.dist(lambdas[(i*3)+0], lambdas[(i*3)+2])) / dim
            E = np.array([Eab, Ebc, Eac])
            coef = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
            thetas_abc = np.reciprocal(np.linalg.solve(coef, E))
            self.thetas += list(thetas_abc)

    def predict(self, lambdas, max_steps=200, lr=0.001, 
            verbose=False, vote=False):
        n = lambdas[0].shape[0]
        ys = torch.zeros_like(lambdas[0])
        for i in tqdm(range(n)):
            if vote:
                thetas = np.ones_like(self.thetas)
            else:
                thetas = self.thetas
                
            # Initialize y to be on the manifold
            y = torch.randn_like(lambdas[0][0])
            y = torch.autograd.Variable(man.projx(y), requires_grad=True) 
            
            for step in range(max_steps):
                objective = torch.zeros(1)
                for j in range(self.num_lfs):
                    objective += (
                        thetas[j] * man.dist2(lambdas[j][i], y))
                objective.backward()

                # Riemannian descent step
                with torch.no_grad():
                    dy = man.egrad2rgrad(y, y.grad)
                    ynew = man.expmap(y, -lr * dy)
                    y.zero_()
                    y.add_(ynew)
                y.grad.zero_()
            
                if verbose:
                    print(i, step, objective.item())
            
            ys[i] += y
        return ys
