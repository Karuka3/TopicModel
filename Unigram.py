import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma
from tqdm import tqdm

plt.rcParams["font.family"] = 'MS Gothic'
plt.style.use("ggplot")


class Unigram:

    def __init__(self, wordset, beta=2):
        wordset = pd.Series(wordset)
        self.beta = beta
        self.Nv = wordset.value_counts()
        self.V = len(self.Nv)
        self.N = self.Nv.sum()

    def fit(self, estimator="ML"):
        if estimator == "ML":
            self.phi = self.Nv / self.N
        elif estimator == "MAP":
            self.phi = (self.Nv + self.beta - 1) / \
                (self.N + (self.beta - 1) * (self.V))
        # elif estimator == "Bayes":
        #    self.phi = (self.Nv + self.beta) / (self.N + (self.beta * self.V))
        else:
            print("Estimator is ML or MAP")

    def gen_words(self, n=20, random_state=21):
        np.random.seed(random_state)
        for i in range(n):
            word = np.random.choice(self.phi.index, p=self.phi)
            print(word)

    def graph(self, n=30, max_x=0.005, save=False):
        fig = plt.barh(range(len(self.phi))[:n], self.phi.iloc[:n][::-1])
        plt.xlim(0, max_x)
        plt.ylim(0, n)
        plt.xlabel(r"Prob")
        plt.ylabel(r"word")
        plt.title(r"Phi")
        plt.yticks(range(len(self.phi))[:n],
                   self.phi.index[:n][::-1], rotation=0)
        plt.show()
        if save:
            fig.save("Unigram.png")

    def estimate_hyper_param(self, beta=10, max_iter=100):
        """
        Empirical Bayesian Estimation
        """
        for i in range(max_iter):
            numerator = digamma(self.Nv + beta).sum() - self.V * digamma(beta)
            denominator = self.V * \
                digamma(self.N + beta * self.V) - \
                self.V * digamma(beta * self.V)
            beta_new = beta * numerator / denominator
            if np.linalg.norm(beta - beta_new) / np.linalg.norm(beta) < 0.01:
                print("beta is converged!\nbeta: {}".format(beta))
                self.beta = beta
                break
            beta = beta_new
            if i == max_iter - 1:
                print("beta is not converged")


class MixtureUnigram:
    def __init__(self, K=9, alpha=0.1, beta=0.1, random_state=21):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def fit(self, docs, word2num, bows, estimator="ML", max_iter=1000):
        self.W = docs
        self.D = len(docs)
        self.V = len(word2num)
        self.Nd = [len(d) for d in self.W]
        self.Ndv = bows
        self.theta, self.phi, self.q = self.init_params()
        self.word2num = word2num

        if estimator == "ML":
            for iter in tqdm(range(max_iter)):
                theta_new = np.zeros(self.K)
                phi_new = np.zeros([self.K, self.V])
                for d in range(self.D):
                    for k in range(self.K):
                        self.q[d][k] = self.e_step(d, k)
                        theta_new, phi_new = self.m_step(
                            d, k, theta_new, phi_new)
                self.theta = self.normarization(theta_new)
                self.phi = self.normarization(phi_new, axis=0)
            save = {"theta": self.theta, "phi": self.phi}
            return save

    def init_params(self):
        np.random.seed(self.random_state)
        theta = np.random.uniform(0, 1, self.K)
        theta /= theta.sum()
        phi = np.random.uniform(0, 1, [self.K, self.V])
        phi /= phi.sum(axis=1)[:, np.newaxis]
        q = np.zeros([self.D, self.K])
        return theta, phi, q

    def e_step(self, d, k):
        numerator = self.theta[k]
        for v in range(self.V):
            numerator *= self.phi[k][v]**self.Ndv[d][v]
        denominator = []
        for k_ in range(self.K):
            for v in range(self.V):
                tmp = self.phi[k_][v]**self.Ndv[d][v]
            denominator.append(self.theta[k_] * tmp)
        q = numerator / np.sum(denominator)
        return q

    def m_step(self, d, k, theta_new, phi_new):
        theta_new[k] += self.q[d][k]
        for n in range(self.Nd[d]):
            v = self.word2num[self.W[d][n]]
            phi_new[k][v] += self.q[d][k]
        return theta_new, phi_new

    def normarization(self, ndarray, axis=0):
        ndarray /= ndarray.sum(axis=axis)
        return ndarray
