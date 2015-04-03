
from qmp.utilities import *
import numpy as np


class Potential3D(object):

    def __init__(self, cell=None, f=None, firstd=None, secondd=None):

        if cell is None:
            cell = [[0.,1.],[0.,1.],[0.,1.]]
        self.cell = cell
        self.f = f
        self.firstd = firstd
        self.secondd = secondd
        self.data = None

    def __call__(self,x,y,z):

        if self.f is None:
            return np.zeros((len(x),len(y), len(z)))
        else:
            return self.f(x,y,z)

    def deriv(self,x,y,z):

        if self.firstd is None:
            return num_deriv_2D(self.f,x,y,z)
        else:
            return self.firstd(x,y,z)

    def hess(self,x,y,z):

        if self.secondd is None:
            return num_deriv2_2D(self.f,x,y,z)
        else:
            return self.secondd(x,y,z)


    def plot_pot(self,start=[0.,0.,0.], end=[1.,1.,1.], pts=50):

        try:
            from matplotlib import pyplot as plt
        except:
            raise ImportError('cannot import matplotlib')

        x, y, z = np.linspace(start[0], end[0], pts), np.linspace(start[1], end[1], pts), \
                  np.linspace(start[2], end[2], pts)
        xgrid, ygrid, zgrid = np.meshgrid(x,y,z)

        V = self.f(xgrid,ygrid,zgrid)

        plt.imshow(V)

