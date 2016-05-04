
from qmp.tools.utilities import *
import numpy as np


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
        except:
            raise ImportError('cannot import matplotlib')

        x, y = np.linspace(start[0], end[0], pts), np.linspace(start[1], end[1], pts)
        xgrid, ygrid = np.meshgrid(x,y)

        V = self.f(xgrid,ygrid)

        plt.imshow(V)

