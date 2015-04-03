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


def num_deriv_2D(f, x,y, incr_x=0.01,incr_y=0.01):
    return ((f(x+incr_x,y)-f(x-incr_x,y))/2./incr_x) + \
           ((f(x,y+incr_y)-f(x,y-incr_y))/2./incr_y)


def num_deriv2_2D(f, x,y, incr_x=0.01,incr_y=0.01):
    return (f(x+incr_x,y)-2.*f(x,y)+f(x-incr_x,y))/(incr_x*incr_x) + \
           (f(x,y+incr_y)-2.*f(x,y)+f(x,y-incr_y))/(incr_y*incr_y)
