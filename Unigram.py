import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma
from tqdm import tqdm

plt.rcParams["font.family"] = 'MS Gothic'
plt.style.use("ggplot")


class Unigram:
    """
    D:      the number of docuements
    Nd:     the number of words in d
    V:      the number of vocabulary
    W:      Document set
    wd:     word set in d
    d:      document index
    v:      vocabulary index
    n:      word index
    N:      total word counts
    Nv:     vocaburaly v counts in all documents
    Ndv:    vocaburaly v counts in d
    """

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
