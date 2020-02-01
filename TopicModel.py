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
