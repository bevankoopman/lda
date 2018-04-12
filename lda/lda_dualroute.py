# coding=utf-8
"""
Julia implementation of the Dual Route Topic Model:

C. Chemudugunta, P. Smyth, and M. Steyvers. Modeling general and specific aspects of documents with a probabilistic topic model.
In Advances in neural information processing systems, pages 241–248, 2007.

"""


import logging, random
import numpy as np

from lda import *

logger = logging.getLogger('lda_dualroute')

class LDADualRoute(LDA):

    def __init__(self, n_topics, n_iter=2000, beta=0.01):


        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = 50 / n_topics
        self.beta = beta

        if self.alpha <= 0 or beta <= 0:
            raise ValueError("alpha and eta must be greater than zero")


    def _fit(self, pwz):
        """Fit the model to the data X"""

        self._initialize(X)

        return self

    def _dualroute2(self, wind, pwz):

        N = len(wind) # number of terms in the corpus
        n_topics = self.n_topics # number of topics passed in as param
        n_iter = self.n_iter # number of iterations for the Gibbs sampling



        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc) # number of words assigned to each topic
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc) # number of documents assigned to each topic
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc) # number of times each topics is assinged

        self.x = np.zeros(N, dtype=np.intc) # assign the words to routes (x)
        self.nr = np.zeros(2, dtype=np.intc) # number of words assigned to each route

        self.z = np.zeros(N, dtype=np.intc)  # assign the words to topics (z)
        self.ztot = np.zeros(n_topics, dtype=np.intc) # total number words assigned to each topic

        self.nv = np.zeros(N, dtype=np.intc) # number of words assigned to verbatim route

        for w_i in N:
            random_route = random.choice([0, 1]) # 0 (verbatim) or 1 (gist)
            self.x[w_i] = random_route # assign this would to the random route
            self.nr[random_route] += 1

            random_topic = random.choice(range(n_topics))
            self.z[w_i] = random_route
            self.ztot[random_topic] += 1

            if random_route == 0:
                self.nv[w_i] += 1

        sumnv = sum(self.nv)


        for iteration in range(1, n_iter+1):

            words = list(range(1,N+1))
            random.shuffle(words)

            for w_i in words:

                # get current assignments
                cz = self.z[w_i]
                cx = self.x[w_i]

                # subtract current assignments
                self.nr[cx] -= 1
                self.ztot[cz] -= 1

                if cx = 0:
                    self.nv[w_i] -= 1
                    sumnv -= 1


