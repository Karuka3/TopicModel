import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'MS Gothic'
plt.style.use("ggplot")


class PLSA:

    def __init__(self, K=9, random_state=17):
        self.K = K
        self.random_state = random_state

    def fit(self, docs, word2num, bows, max_iter=1000):
        self.W = docs
        self.D = len(docs)
        self.V = len(word2num)
        self.Nd = [len(d) for d in self.W]
        self.Ndv = bows
        self.theta, self.phi, self.q = self.init_params()
        self.word2num = word2num

        for __iter in tqdm(range(max_iter)):
            theta_new = np.zeros([self.D, self.K])
            phi_new = np.zeros([self.K, self.V])
            for d in range(self.D):
                for n in range(len(self.W[d])):
                    for k in range(self.K):
                        self.q[d][k][n] = self.e_step(d, n, k)
                        theta_new, phi_new = self.m_step(
                            d, n, k, theta_new, phi_new)
            self.theta /= self.theta.sum(axis=1)[:, np.newaxis]
            self.phi /= self.phi.sum(axis=1)[:, np.newaxis]
            self.phi[self.phi < 1e-100] = 1e-100

    def init_params(self):
        np.random.seed(self.random_state)
        theta = np.random.rand(self.D, self.K)
        phi = np.random.rand(self.K, self.V)
        theta /= theta.sum(axis=1)[:, np.newaxis]
        phi /= phi.sum(axis=1)[:, np.newaxis]
        q = [[[0] * n] * self.K for n in self.Nd]
        return theta, phi, q

    def e_step(self, d, n, k):
        if type(self.W[d][n]) is str:
            v = self.word2num[self.W[d][n]]
        else:
            v = self.W[d][n]
        numerator = self.theta[d][k] * self.phi[k][v]
        denominator = 0
        for k_ in range(self.K):
            denominator += self.theta[d][k_] * self.phi[k_][v]
        q = numerator / denominator
        return q

    def m_step(self, d, n, k, theta_new, phi_new):
        theta_new[d][k] += self.q[d][k][n]
        if type(self.W[d][n]) is str:
            for n_ in range(self.Nd[d]):
                v = self.word2num[self.W[d][n_]]
                phi_new[k][v] += self.q[d][k][n_]
        else:
            for n_ in range(self.Nd[d]):
                v = self.W[d][n_]
                phi_new[k][v] += self.q[d][k][n_]
        return theta_new, phi_new

    def graph_phi(self, num2word, d=0, n=30, max_x=0.0005):
        topic = self.theta[d].argmax()
        word_prob = pd.Series(self.phi[topic])
        word_prob.index = [num2word[v] for v in range(self.V)]
        phi_d = word_prob.sort_values()[::-1][:n]
        print(word_prob)
        phi_d_position = np.arange(len(phi_d[:n]))
        plt.barh(phi_d_position, phi_d)
        plt.xlim(0, max_x)
        plt.ylim(0, n)
        plt.xlabel(u"Prob")
        plt.ylabel(u"word")
        plt.title(u"Phi Topic{} ".format(topic))
        plt.yticks(phi_d_position, phi_d.index[:n][::-1], rotation=0)
        plt.show()

    def graph_theta(self, d=0):
        plt.bar(np.arange(self.K), self.theta[d], align="center", color="red")
        plt.title(u"Document {}\n θ ".format(d))
        plt.xlabel(u"Topic")
        plt.ylabel(u"Prob")
        plt.xticks(np.arange(self.K), np.arange(self.K))
        plt.show()

    def make_topic_distribution(self, N=30):
        phi_df = pd.DataFrame(self.phi.T, index=self.word2num.keys())
        topic_df = pd.DataFrame(index=range(N), columns=range(self.K))
        for k in range(self.K):
            topic_df.iloc[:, k] = phi_df[k].sort_values()[
                ::-1][:N].index.values
        print(topic_df)


class LDA:
    def __init__(self, K=9, alpha=0.1, beta=0.1, random_state=717):
        self.K = K          # <- topic count
        self.alpha = alpha  # <- parameter of topics prior
        self.beta = beta    # <- parameter of words proir
        self.random_state = random_state

    def fit(self, docs, word2num, estimator="Gibbs", max_iter=1000):
        np.random.seed(self.random_state)
        self.W = docs
        self.V = len(word2num)  # <- word count
        self.D = len(docs)      # <- documents count
        self.Z = self.init_topics()
        self.ndk, self.nkv, self.nd, self.nk = self.init_params()

        for __iter in tqdm(range(max_iter)):
            for i, d in enumerate(self.Z):
                for j, k in enumerate(d):
                    v = self.W[i][j]
                    self.ndk[i, k] -= 1
                    self.nkv[k, v] -= 1
                    self.nk[k] -= 1
                    new_z = self.sampling(i, v)
                    # update topics
                    self.Z[i][j] = new_z
                    self.ndk[i, new_z] += 1
                    self.nkv[new_z, v] += 1
                    self.nk[new_z] += 1
        save = {"topics": self.Z, "nkv": self.nkv,
                "ndk": self.ndk, "nk": self.nk, "nd": self.nd}
        return save

    def sampling(self, i, v):
        probs = np.zeros(self.K)
        for k in range(self.K):
            theta = (self.ndk[i, k] + self.alpha) / \
                (self.nd[i] + self.alpha * self.D)
            phi = (self.nkv[k, v] + self.beta) / \
                (self.nk[k] + self.beta * self.K)
            prob = theta * phi
            probs[k] = prob
        probs /= probs.sum()
        z = np.where(np.random.multinomial(1, probs) == 1)[0][0]
        return z

    def init_topics(self):
        topics = [[np.random.randint(self.K) for w in d] for d in self.W]
        return topics

    def init_params(self):
        ndk = np.zeros((self.D, self.K))  # <- topic distribution of sentences
        nkv = np.zeros((self.K, self.V))  # <- word disrtibution of each topics
        for i, d in enumerate(self.Z):
            for j, z in enumerate(d):
                ndk[i, z] += 1
                nkv[z, self.W[i][j]] += 1
        nd = ndk.sum(axis=1)
        nk = nkv.sum(axis=1)
        return ndk, nkv, nd, nk


class JointTopicModel:
    def __init__(self, K, alpha, beta, random_state=717):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        np.random.seed(random_state)

    def fit(self, docs, word2num, category, max_iter=1000):
        self.W = docs
        self.T = len(docs)
        self.D = len(docs[0])
        self.V = [len(word2num), len(category)]
        self.Z = self.init_topics()
        self.ndk, self.nkv, self.nk, self.nd = self.init_params()

        for __iter in tqdm(range(max_iter)):
            for t in range(self.T):
                for d in np.random.choice(self.D, self.D, replace=False):
                    for i in np.random.choice(len(self.W[t][d]), len(self.W[t][d]), replace=False):
                        k = self.Z[t][d][i]
                        v = self.W[t][d][i]

                        self.ndk[t][d][k] -= 1
                        self.nkv[t][k][v] -= 1
                        self.nk[t][k] -= 1

                        new_z = self.sampling(t, d, v, self.nk[t])
                        self.Z[t][d][i] = new_z
                        self.ndk[t][d][new_z] += 1
                        self.nkv[t][new_z][v] += 1
                        self.nk[t][new_z] += 1

        save = {"topics": self.Z, "nkv": self.nkv,
                "ndk": self.ndk, "nk": self.nk, "nd": self.nd}
        return save

    def init_topics(self):
        Z = {}
        for t in range(self.T):
            Z[t] = [np.random.randint(low=0, high=self.K, size=len(
                self.W[t][d])) for d in range(len(self.W[t]))]
        return Z

    def init_params(self):
        ndk = {}
        nkv = {}
        nk = {}
        nd = {}
        for t in range(self.T):
            ndk[t] = np.zeros((self.D, self.K)) + self.alpha
            nkv[t] = np.zeros((self.K, self.V[t])) + self.beta
            for d in range(self.D):
                for i in range(len(self.W[t][d])):
                    k = self.Z[t][d][i]
                    v = self.W[t][d][i]
                    ndk[t][d, k] += 1
                    nkv[t][k, v] += 1
            nk[t] = nkv[t].sum(axis=1)
            nd[t] = ndk[t].sum(axis=1)
        return ndk, nkv, nk, nd

    def sampling(self, t, d, v, nk):
        nkv = self.nkv[t][:, v]  # k-dimensional vector
        prob = (sum([self.ndk[t][d] for t in range(self.T)]) -
                self.alpha*(self.T-1)) * (nkv/nk)
        prob = prob/prob.sum()
        z = np.random.multinomial(n=1, pvals=prob).argmax()
        return z
