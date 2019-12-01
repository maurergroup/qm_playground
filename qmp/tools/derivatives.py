#    qmp.tools.derivatives
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
Functions for calculating various derivatives.
"""
import numpy as np


def num_deriv(f, x, i=0, j=0, incr=0.001):
    """Calculates first numerical derivative of f on point x."""
    deriv = (-f(x+2.*incr, i=i, j=j)
             + 8.*f(x+incr, i=i, j=j)
             - 8.*f(x-incr, i=i, j=j)
             + f(x-2.*incr, i=i, j=j))
    return deriv / (12*incr)


def num_deriv2(f, x, i=0, j=0, incr=0.001):
    """Calculates second numerical derivative of f on point x."""
    hess = (-f(x+2.*incr, i=i, j=j)
            + 16.*f(x+incr, i=i, j=j)
            - 30.*f(x, i=i, j=j)
            + 16.*f(x-incr, i=i, j=j)
            - f(x-2.*incr, i=i, j=j))
    return hess/(12.*incr**2)


def num_deriv3(f, x, incr=0.001):
    """Return third derivative of f at x."""
    return (-f(x+3.*incr)+8.*f(x+2.*incr)-13.*f(x+incr)+13.*f(x-incr)
            - 8.*f(x-2.*incr)+f(x-3.*incr))/(8.*incr**3)


def num_deriv4(f, x, incr=0.001):
    """Return fourth derivative of f at x."""
    return (-f(x+3.*incr)+12.*f(x+2.*incr)-39.*f(x+incr)+56.*f(x)
            - 39.*f(x-incr)+12.*f(x-2.*incr)-f(x-3.*incr))/(6.*incr**4)


def num_deriv_2D(f, x, y, i=0, j=0, incr_x=0.001, incr_y=0.001):
    """Return first Nabla*f"""
    return np.array([(f(x+incr_x, y, i=i, j=j)
                     - f(x-incr_x, y, i=i, j=j))/2./incr_x,
                     (f(x, y+incr_y, i=i, j=j)
                     - f(x, y-incr_y, i=i, j=j))/2./incr_y])


def num_deriv2_2D(f, x, y, i=0, j=0, incr_x=0.001, incr_y=0.001):
    """Return < Nabla^2 | f >"""
    return ((f(x+incr_x, y, i=i, j=j)
            - 2.*f(x, y, i=i, j=j)
            + f(x-incr_x, y, i=i, j=j))/(incr_x*incr_x)
            + (f(x, y+incr_y, i=i, j=j)
            - 2.*f(x, y, i=i, j=j)
            + f(x, y-incr_y, i=i, j=j))/(incr_y*incr_y))
