import numpy as np
from tqdm import tqdm


class LDA:
    def __init__(self, K=5, alpha=0.1, beta=0.1, verbose=0, random_state=17):
        self.K = K          # <- topic count
        self.alpha = alpha  # <- parameter of topics prior
        self.beta = beta    # <- parameter of words proir
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, docs, word2num, max_iter=1000):
        np.random.seed(self.random_state)
        self.docs = docs
        self.V = len(word2num)  # <- word count
        self.M = len(docs)      # <- documents count
        self.topics = self.init_topics()
        self.ndk, self.nkv, self.nd, self.nk = self.init_params()

        for iter in tqdm(range(max_iter)):
            for i, d in enumerate(self.topics):
                for j, k in enumerate(d):
                    v = self.docs[i][j]
                    self.ndk[i, k] -= 1
                    self.nkv[k, v] -= 1
                    self.nk[k] -= 1
                    new_z = self.sampling(i, v)
                    # update topics
                    self.topics[i][j] = new_z
                    self.ndk[i, new_z] += 1
                    self.nkv[new_z, v] += 1
                    self.nk[new_z] += 1
        save = {"topics": self.topics, "nkv": self.nkv,
                "ndk": self.ndk, "nk": self.nk, "nd": self.nd}
        return save

    def sampling(self, i, v):
        probs = np.zeros(self.K)
        for k in range(self.K):
            theta = (self.ndk[i, k] + self.alpha) / \
                (self.nd[i] + self.alpha * self.M)
            phi = (self.nkv[k, v] + self.beta) / \
                (self.nk[k] + self.beta * self.K)
            prob = theta * phi
            probs[k] = prob
        probs /= probs.sum()
        z = np.where(np.random.multinomial(1, probs) == 1)[0][0]
        return z

    def init_topics(self):
        topics = [[np.random.randint(self.K) for w in d] for d in self.docs]
        return topics

    def init_params(self):
        ndk = np.zeros((self.M, self.K))  # <- topic distribution of sentences
        nkv = np.zeros((self.K, self.V))  # <- word disrtibution of each topics
        for i, d in enumerate(self.topics):
            for j, z in enumerate(d):
                ndk[i, z] += 1
                nkv[z, self.docs[i][j]] += 1
        nd = ndk.sum(axis=1)
        nk = nkv.sum(axis=1)
        return ndk, nkv, nd, nk


class JointTopicModel:
    def __init__(self, K, alpha, beta, max_iter, verbose=0):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, V):
        self._X = X
        self._T = len(X)  # number of vocab types
        self._N = len(X[0])  # number of documents
        self._V = V  # number of vocabularies for each t
        self.Z = self._init_topics()
        self.ndk, self.nkv = self._init_params()
        nk = {}
        for t in range(self._T):
            nk[t] = self.nkv[t].sum(axis=1)

        remained_iter = self.max_iter
        while True:
            if self.verbose:
                print(remained_iter)
            for t in np.random.choice(self._T, self._T, replace=False):
                for d in np.random.choice(self._N, self._N, replace=False):
                    for i in np.random.choice(len(self._X[t][d]), len(self._X[t][d]), replace=False):
                        k = self.Z[t][d][i]
                        v = self._X[t][d][i]

                        self.ndk[t][d][k] -= 1
                        self.nkv[t][k][v] -= 1
                        nk[t][k] -= 1

                        self.Z[t][d][i] = self._sample_z(t, d, v, nk[t])
                        self.ndk[t][d][self.Z[t][d][i]] += 1
                        self.nkv[t][self.Z[t][d][i]][v] += 1
                        nk[t][self.Z[t][d][i]] += 1
            remained_iter -= 1
            if remained_iter <= 0:
                break
        return self

    def _init_topics(self):
        Z = {}
        for t in range(self._T):
            Z[t] = []
            for d in range(len(self._X[t])):
                Z[t].append(np.random.randint(
                    low=0, high=self.K, size=len(self._X[t][d])))
        return Z

    def _init_params(self):
        ndk = {}
        nkv = {}
        for t in range(self._T):
            ndk[t] = np.zeros((self._N, self.K)) + self.alpha
            nkv[t] = np.zeros((self.K, self._V[t])) + self.beta
            for d in range(self._N):
                for i in range(len(self._X[t][d])):
                    k = self.Z[t][d][i]
                    v = self._X[t][d][i]
                    ndk[t][d, k] += 1
                    nkv[t][k, v] += 1
        return ndk, nkv

    def _sample_z(self, t, d, v, nk):
        nkv = self.nkv[t][:, v]  # k-dimensional vector
        prob = (sum([self.ndk[t][d] for t in range(self._T)]) -
                self.alpha*(self._T-1)) * (nkv/nk)
        prob = prob/prob.sum()
        z = np.random.multinomial(n=1, pvals=prob).argmax()
        return z
