"""

"""

from qmp.utilities import *
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


class Potential2D(object):

    def __init__(self, cell=None, f=None, firstd=None, secondd=None):

        if cell is None:
            cell = [[0.,1.],[0.,1.]]
        self.cell = cell
        self.f = f
        self.firstd = firstd
        self.secondd = secondd
        self.data = None

    def __call__(self,x,y):

        if self.f is None:
            return np.zeros((len(x),len(y)))
        else:
            return self.f(x,y)

    def deriv(self,x,y):

        if self.firstd is None:
            return num_deriv_2D(self.f,x,y)
        else:
            return self.firstd(x,y)

    def hess(self,x,y):

        if self.secondd is None:
            return num_deriv2_2D(self.f,x,y)
        else:
            return self.secondd(x,y)


    def plot_pot(self,start=[0.,0.], end=[1.,1.], pts=50):

        try:
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except:
            raise ImportError('cannot import matplotlib or mpl_toolkits')

        x, y = np.linspace(start[0], end[0], pts), np.linspace(start[1], end[1], pts)
        xgrid, ygrid = np.meshgrid(x,y)

        V = self.f(xgrid,ygrid)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xgrid, ygrid, V, alpha=.5)
        plt.show()
    

