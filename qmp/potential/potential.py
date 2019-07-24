#qmp.potential.potential
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
"""
class potential
"""

from qmp.tools import utilities
import numpy as np


class Potential(object):
    """
    Defines Potential and all operations on it. Functions for the
    potential energy, the first and second derivative, and for possible
    excited states and scalar and vectorial derivative couplings must
    be passed explicitly.
    """

    def __init__(self, cell=[[0., 1.]], f=lambda a: 0, firstd=None, n=1,
                 secondd=None, d1=None, d2=None):
        """
        Initializes potential
        Input
            cell: domain given as list of two values
            f:    function that defines potential
            firstd: first derivative, if not given is calculated numerically
            secondd: second derivative, if not given is calculated numerically
            n: number of states, default is 1. if n>1, f,d1,d2,firstd, secondd are lists of functions
            d1: first order couplings
            d2: second order couplings
        """

        self.cell = cell
        self.dimension = len(self.cell)

        if not isinstance(f, list):
            f = [f]
        self.f = f

        self.states = n

        if not isinstance(firstd, list):
            firstd = [firstd]
        self.firstd = firstd

        if not isinstance(secondd, list):
            secondd = [secondd]
        self.secondd = secondd

        if not isinstance(d1, list):
            d1 = [d1]
        self.d1 = d1

        if not isinstance(d2, list):
            d2 = [d2]
        self.d2 = d2

        self.data = None

    def __call__(self, *points, n=0):
        """
        calculate potential at point x or list of points x
        """
        f = self.f[n]
        return f(*points)

    def deriv(self, points, n=0):
        """
        calculate 1st derivative at point x or list of points x
        """
        if not isinstance(points, list):
            return self.single_point_deriv(points, n=n)

        result = np.zeros(np.shape(points))
        for i, point in enumerate(points):
            try:
                assert len(point) == self.dimension
            except AssertionError as error:
                print(error)
            except TypeError:
                print("Dimension of point does not match potential.")
            result[i] = self.single_point_deriv(point, n=n)

        return result

    def single_point_deriv(self, *point, n=0):
        """
        calculate 1st derivative at single point
        """
        firstd = self.firstd[n]
        if firstd is None:
            return utilities.general_deriv(self.f[n], *point, order=1)
        else:
            return firstd(*point)

    def hess(self, points, n=0):
        """
        calculate 2nd derivative at point x or list of points x
        """
        result = np.zeros(len(points))

        for i, point in enumerate(points):
            try:
                assert len(point) == self.dimension
            except AssertionError as error:
                print(error)
            except TypeError:
                print("Dimension of point does not match potential.")
            result[i] = self.single_point_hess(point, n=n)

        return result

    def single_point_hess(self, point, n=0):
        """
        calculate 2nd derivative at single point
        """
        secondd = self.secondd[n]
        if secondd is None:
            return np.sum(utilities.general_deriv(self.f[n], point, order=2))
        else:
            return secondd(*point)

    #### coupling is different for different PES pairs
    #def coupling_d1(self,x,n=0, m=1):
        #"""
        #calculate nonadiabatic vectorial 1st order coupling
        #"""
        #d1 = self.d1[n]
        #if d1 is None:
            #return np.zeros_like(x)
        #else:
            #return d1(x)

    #def coupling_d2(self,x,n=0, m=1):
        #"""
        #calculate nonadiabatic vectorial 1st order coupling
        #"""
        #d2 = self.d2[n,m]
        #if d2 is None:
            #return np.zeros_like(x)
        #else:
            #return d2(x)

    def plot_pot(self, pts=50):
        """
        plot potential with matplotlib
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError('cannot import matplotlib')

        x = np.linspace(self.cell[0][0], self.cell[0][1], pts)
        y = self.f[0](x)

        plt.plot(x, y)

    def plot_2d(self):
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError('cannot import matplotlib')

        x = np.linspace(self.cell[0][0], self.cell[0][1], 50)
        y = np.linspace(self.cell[1][0], self.cell[1][1], 50)

        X, Y = np.meshgrid(x, y)
        Z = self(X, Y)

        plt.contourf(X, Y, Z, cmap='RdGy')
