"""
utilities including
constants
functions for numerical derivatives (1D: order incr^4)
"""

hbar = 1.
mass = 1.
kB = 3.16681E-6

import numpy as np


def num_deriv(f,x,incr=0.001):
    """
    calculates first numerical derivative
    of f on point x
    """
    return (-f(x+2.*incr) + 8.*f(x+incr) - 8.*f(x-incr) + f(x-2.*incr))/(12.*incr)

def num_deriv2(f,x,incr=0.001):
    """
    calculates second numerical derivative
    of f on point x
    """
    hess = -f(x+2.*incr) + 16.*f(x+incr) - 30.*f(x) + 16.*f(x-incr) - f(x-2.*incr)
    return hess/(12.*incr**2)


def num_deriv3(f,x,incr=0.001):
    """
    return third derivative of f at x
    """
    return (-f(x+3.*incr)+8.*f(x+2.*incr)-13.*f(x+incr)+13.*f(x-incr)-8.*f(x-2.*incr)+f(x-3.*incr))/(8.*incr**3)


def num_deriv4(f,x,incr=0.001):
    """
    return fourth derivative of f at x
    """
    return (-f(x+3.*incr)+12.*f(x+2.*incr)-39.*f(x+incr)+56.*f(x)-39.*f(x-incr)+12.*f(x-2.*incr)-f(x-3.*incr))/(6.*incr**4)


def num_deriv_2D(f, x,y, incr_x=0.001,incr_y=0.001):
    """
    returns first Nabla*f
    """
    return np.array([(f(x+incr_x,y)-f(x-incr_x,y))/2./incr_x, \
                    (f(x,y+incr_y)-f(x,y-incr_y))/2./incr_y])


def num_deriv2_2D(f, x,y, incr_x=0.001,incr_y=0.001):
    """
    returns < Nabla^2 | f >
    """
    return (f(x+incr_x,y)-2.*f(x,y)+f(x-incr_x,y))/(incr_x*incr_x) + \
           (f(x,y+incr_y)-2.*f(x,y)+f(x,y-incr_y))/(incr_y*incr_y)
