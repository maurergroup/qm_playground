"""
solver.py
Different solvers for time-independent 
problems
"""

from qmp.utilities import *
import numpy as np
import scipy as sp


class solver(object):
    """
    Base class for all solver
    """

    def __init__(self, data=None, potential=None):
        
        self.data = data
        self.pot = potential

