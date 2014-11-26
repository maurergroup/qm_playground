"""
data_containers.py
collection of different data containers for different 
jobs such as wavefunction, particle, RPMD necklace
"""


class data_container(object):
    """
    Base class for all data containers
    """

    def __init__(self):

        #dimensions

        self.ndim = None
        self.mass = None
        self.cell = None


class wave_container(data_container):
    """
    Data Container for wavefunction calculations
    """

    def __init__(self):
        """
        Initialises important data structure for wvfn.
        """

        data_container.__init__(self)
        
        #Eigenvectors and Eigenvalues
        self.psi = None
        self.E = None


#class traj_container(data_container):
    #"""

    #"""


#class rpmd_container(data_container):

data_containers= {
    'wave': wave_container,
    #'traj': traj_container,
    #'rpmd': rpmd_container,
        }

