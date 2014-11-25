"""

"""

hbar = 1.0
mass = 1




def num_deriv(f,x,incr=0.01):
    """
    calculates first numerical derivative
    of f on point x
    """

    return (f(x+incr) - f(x-incr))/(2.*incr)

def num_deriv2(f,x,incr=0.01):
    """
    calculates first numerical derivative
    of f on point x
    """
    
    hess = f(x+incr) - 2*f(x) + f(x-incr)
    return hess/(incr*incr)
