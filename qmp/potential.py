"""

"""

from utilities import *
import numpy as np

class Potential(object):
    """
    Defines Potential and all operations on it
    """

    def __init__(self, cell=None, f=None, firstd=None, secondd=None):
        """

        """

        if cell is None:
            cell = [[0.,1.]]
        self.cell = cell
        self.f = f
        self.firstd = firstd
        self.secondd = secondd
        self.data = None

    def __call__(self,x):

        if self.f is None:
            return np.zeros_like(x)
        else:
            return self.f(x)
    
    def deriv(self,x):

        if self.firstd is None:
            return num_deriv(self.f,x)
        else:
            return self.firstd(x)

    def hess(self,x):

        if self.secondd is None:
            return num_deriv2(self.f,x)
        else:
            return self.secondd(x)


    def plot_pot(self,start=0.0, end=1.0, pts=50):

        try:
            from matplotlib import pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        L = np.linspace(start, end, pts)
        y = np.zeros_like(L)
        for i,x in enumerate(L):
            y[i]=self.f(x)

        plt.plot(L, y)
