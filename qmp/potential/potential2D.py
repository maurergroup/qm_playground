#qmp.potential.potential2D
#
#    qm_playground - python package for dynamics simulations
#    Copyright (C) 2016  Reinhard J. Maurer 
#
#    This file is part of qm_playground.
#    
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>#

from qmp.tools.utilities import *
import numpy as np


class Potential2D(object):

    def __init__(self, domain=None, f=None, firstd=None, secondd=None):

        if domain is None:
            domain = [[0.,1.],[0.,1.]]
        self.domain = domain
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

