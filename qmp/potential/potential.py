#    qmp.potential.potential
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
This module contains the Potential class.
"""

import numpy as np

from qmp.tools import derivatives


class Potential:
    """
    Defines Potential and all operations on it.
    """

    def __init__(self, cell, f, n=1, firstd=None,
                 secondd=None, d1=None, d2=None):
        """Initialises a Potential object.

        Initialisation consists of reading the input and ensuring arrays are
        reshaped correctly.

        Parameters
        ----------
        cell : 2-D array
            Simulation box, given as a 2D array, each row corresponding to the
            start and end values of the box in each dimension.
        f : ndarray
            An array of functions with n by n entries that can be reshaped into
            an array representing the matrix elements of the potential
            operator.
        n : int, optional
            Number of electronic states.
        firstd : ndarray, optional
            First derivative function of the potential. If needed and not given
            it will be calculated numerically.
        secondd : ndarray, optional
            Second derivative function of the potential. If needed and not
            given it will be calculated numerically.
        d1 : ndarray
            First order couplings, currently not used.
        d2 : ndarray
            Second order couplings, currently not used.
        """

        self.cell = cell
        self.dimension = len(self.cell)
        self.states = n

        try:
            self.f = np.array(f).reshape((self.states, self.states))
        except ValueError:
            print('The list of functions given to the potential is not of '
                  + 'the correct size to match the number of states.'
                  + ' You must provide n**2 elements.')
            raise

        if firstd is not None:
            try:
                self.firstd = np.array(firstd).reshape((self.states,
                                                        self.states))
            except ValueError:
                print('The list of functions given for derivative is not of '
                      + 'the correct size to match the number of states.'
                      + ' You must provide n**2 elements.')
                raise
        else:
            self.firstd = firstd

        if secondd is not None:
            try:
                self.secondd = np.array(secondd).reshape((self.states,
                                                          self.states))
            except ValueError:
                print('The list of functions given for hessian is not of '
                      + 'the correct size to match the number of states.'
                      + ' You must provide n**2 elements.')
                raise
        else:
            self.secondd = secondd

        # if not isinstance(d1, list):
        #     d1 = [d1]
        # self.d1 = d1

        # if not isinstance(d2, list):
        #     d2 = [d2]
        # self.d2 = d2

    def __call__(self, *points, i=0, j=0):
        """Calculate the potential at the point(s) given.

        The potential is evaluated for the matrix element [i, j].

        Parameters
        ----------
        points : array_like
            Set of coordinates where the potential is to be evaluated.
        i : int
            Matrix index.
        j : int
            Matrix index.
        """
        f = self.f[i, j]
        return f(*points)

    def deriv(self, points, i=0, j=0):
        """Calculate 1st derivative at point(s)."""
        # Need to make these functions n-dimensional, very ugly right now
        if self.firstd is None:
            if self.dimension == 1:
                return derivatives.num_deriv(self, points, i, j)
            elif self.dimension == 2:
                d = np.empty_like(points)
                for k, point in enumerate(points):
                    d[k] = derivatives.num_deriv_2D(self, point[0],
                                                    point[1], i, j)
                return d
        else:
            firstd = self.firstd[i, j]
            if self.dimension == 1:
                return firstd(points)
            elif self.dimension == 2:
                d = np.empty_like(points)
                for k, point in enumerate(points):
                    d[k] = firstd(point[0], point[1])
                return d

    def hess(self, points, i=0, j=0):
        """Calculate 2nd derivative at point(s)."""
        if self.secondd is None:
            if self.dimension == 1:
                return derivatives.num_deriv2(self, points, i, j)
            elif self.dimension == 2:
                return derivatives.num_deriv2_2D(self, points[:, 0],
                                                 points[:, 1], i, j)
        else:
            secondd = self.secondd[i, j]
            return secondd(*points)

    def compute_cell_potential(self, density=100):
        """Calculate the potential on a grid over the whole cell."""

        if self.dimension == 1:
            x = np.linspace(self.cell[0][0], self.cell[0][1], density)
            out = self(x)

        elif self.dimension == 2:
            x = np.linspace(self.cell[0][0], self.cell[0][1], density)
            y = np.linspace(self.cell[1][0], self.cell[1][1], density)
            xx, yy = np.meshgrid(x, y)
            out = self(xx, yy)

        else:
            raise ValueError(self.dimension+', the dimension, must be 1 or 2.')

        return out
