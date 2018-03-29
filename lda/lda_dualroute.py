# coding=utf-8
"""
Julia implementation of the Dual Route Topic Model:

C. Chemudugunta, P. Smyth, and M. Steyvers. Modeling general and specific aspects of documents with a probabilistic topic model.
In Advances in neural information processing systems, pages 241–248, 2007.

"""

from lda import *

class LDADualRoute(LDA):
    x=0

    def _fit(self, X):
        """Fit the model to the data X"""

        return self