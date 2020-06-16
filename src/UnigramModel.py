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
            self.phi = (self.Nv + self.beta - 1) / (self.N + (self.beta - 1) * (self.V))
        else:
            print("Estimator is ML or MAP")

    def gen_words(self, n=20, random_state=21):
        np.random.seed(random_state)
        for __iter in range(n):
            word = np.random.choice(self.phi.index, p=self.phi)
            print(word)

    def graph(self, n=30, max_x=0.005):
        plt.barh(range(len(self.phi))[:n], self.phi.iloc[:n][::-1])
        plt.xlim(0, max_x)
        plt.ylim(0, n)
        plt.xlabel(u"Prob")
        plt.ylabel(u"word")
        plt.title(u"Phi")
        plt.yticks(range(len(self.phi))[:n], self.phi.index[:n][::-1], rotation=0)
        plt.show()

    def estimate_hyper_param(self, beta=10, max_iter=100):
        """
        Empirical Bayesian Estimation
        """
        for i in range(max_iter):
            numerator = digamma(self.Nv + beta).sum() - self.V * digamma(beta)
            denominator = self.V * digamma(self.N + beta * self.V) - self.V * digamma(beta * self.V)
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
        np.random.seed(random_state)
        self.K = K
        self.alpha0 = alpha
        self.beta0 = beta

    def fit(self, docs, word2num, bows: list, estimator="ML", max_iter=1000):
        self.W = docs
        self.D = len(docs)
        self.V = len(word2num)
        self.Nd = [len(d) for d in self.W]
        self.Ndv = bows
        self.theta, self.phi, self.q, self.alpha, self.beta = self.init_params()
        self.word2num = word2num

        if estimator == "ML":
            for __iter in tqdm(range(max_iter)):
                theta_new = np.zeros(self.K)
                phi_new = np.zeros([self.K, self.V])
                for d in range(self.D):
                    for k in range(self.K):
                        self.q[d][k] = self.e_step(d, k)
                        theta_new, phi_new = self.m_step(d, k, theta_new, phi_new)
                self.theta /= self.theta.sum()
                self.phi /= self.phi.sum(axis=1)[:, np.newaxis]

        elif estimator == "VB":
            for __iter in tqdm(range(max_iter)):
                alpha_new = np.ones(self.K) * self.alpha0
                beta_new = np.ones((self.K, self.V)) * self.beta0
                dig_alpha = digamma(self.alpha) - digamma(self.alpha.sum())
                dig_beta0 = digamma(self.beta)
                dig_beta1 = digamma(self.beta.sum(axis=1, keepdims=True))
                for d in range(self.D):
                    for k in range(self.K):
                        self.q[d][k] = np.exp(dig_alpha[k] + np.dot(self.Ndv[d], dig_beta0[k]) -
                                              self.Nd[d] * dig_beta1[k])
                        alpha_new[k] += self.q[d][k]
                        if type(self.W[d][0]) is str:
                            for n in range(self.Nd[d]):
                                v = self.word2num[self.W[d][n]]
                                beta_new[k][v] += self.q[d][k]
                        else:
                            for n in range(self.Nd[d]):
                                v = self.W[d][n]
                                beta_new[k][v] += self.q[d][k]
                self.alpha = alpha_new
                self.beta = beta_new
            self.theta = np.random.dirichlet(self.alpha)
            self.phi = np.array([np.random.dirichlet(b) for b in self.beta])

    def init_params(self):
        theta = np.random.rand(self.K)
        theta /= theta.sum()
        phi = np.random.rand(self.K, self.V)
        phi /= phi.sum(axis=1)[:, np.newaxis]
        q = np.zeros([self.D, self.K])
        alpha = self.alpha0 + np.random.rand(self.K)
        beta = self.beta0 + np.random.rand(self.K, self.V)
        return theta, phi, q, alpha, beta

    def e_step(self, d, k):
        numerator = self.theta[k]
        denominator = []
        deno = 1
        for v in range(self.V):
            numerator *= self.phi[k][v]**self.Ndv[d][v]
        for k_ in range(self.K):
            for v in range(self.V):
                deno *= self.phi[k_][v]**self.Ndv[d][v]
            denominator.append(self.theta[k_] * deno)
        q = numerator / sum(denominator)
        return q

    def m_step(self, d, k, theta_new, phi_new):
        theta_new[k] += self.q[d][k]
        if type(self.W[d][0]) is str:
            for n in range(self.Nd[d]):
                v = self.word2num[self.W[d][n]]
                phi_new[k][v] += self.q[d][k]
        else:
            for n in range(self.Nd[d]):
                v = self.W[d][n]
                phi_new[k][v] += self.q[d][k]
        return theta_new, phi_new

    def graph_phi(self, num2word, topic=0, n=30, max_x=0.005):
        word_prob = pd.Series(self.phi[topic])
        word_prob.index = [num2word[v] for v in range(self.V)]
        phi_d = word_prob.sort_values(ascending=False)
        plt.barh(np.arange(len(phi_d[:n])), phi_d[:n][::-1])
        plt.xlim(0, max_x)
        plt.ylim(0, n)
        plt.xlabel(u"Prob")
        plt.ylabel(u"word")
        plt.title(u"Phi Topic{} ".format(topic))
        plt.yticks(np.arange(len(phi_d[:n])), phi_d.index[:n][::-1], rotation=0)
        plt.show()

    def graph_theta(self):
        plt.bar(np.arange(self.K), self.theta, align="center", color="red")
        plt.title(u"θ")
        plt.xlabel(u"Topic")
        plt.ylabel(u"Prob")
        plt.xticks(np.arange(self.K), np.arange(self.K))
        plt.show()

    def make_topic_distrtibution(self, N=30):
        phi_df = pd.DataFrame(self.phi.T, index=self.word2num.keys)
        topic_df = pd.DataFrame(index=range(N), columns=range(self.K))
        for k in range(self.K):
            topic_df.iloc[:, k] = phi_df[k].sort_valus()[::-1][:30].index.values()
        return topic_df
