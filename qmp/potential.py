"""

"""

from qmd.utilities import *
import numpy as np

class potential(object):
    """

    """

    def __init__(self, cell, function=None, firstd=None, secondd=None):
        """

        """

        self.cell = cell
        self.f = function
        self.firstd = firstd
        self.secondd = secondd

    def eval(x):

        if self.f is None:
            return 0.0
        else:
            return self.f(x)
    
    def eval_deriv(x):

        if self.firstd is None:
            return num_deriv(self.f,x)
        else:
            return self.firstd(x)

    def eval_hess(x):

        if self.secondd is None:
            return num_deriv2(self.f,x)
        else:
            return self.secondd(x)

    def plot_pot(start=0.0, end=1.0, pts=50):

        try:
            from matplotlib import pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        L = np.linspace(start, end, pts)
        y = np.zeros_like(L)
        for i,x in enumerate(L):
            y[i]=self.f(x)

        plt.plot(L, y)
