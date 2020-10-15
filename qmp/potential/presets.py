"""N-dimensional predefined potential generators.

Functions returned by calling these generators should be given to the Potential
class in order to define the potential of the model."""
import numpy as np
from qmp.tools.dyn_tools import create_real_gaussian2D
from abc import ABC


class ModelPotential(ABC):
    """Abstract base class for the model potential generators."""

    def __init__(self, dimension):
        self.function = None
        self.dimension = dimension

    def __call__(self):
        return self.function


class Free(ModelPotential):
    """Generator for a free potential (no potential)."""

    def __init__(self, dimension):
        ModelPotential.__init__(self, dimension)

        def f_free(*point):
            return np.zeros_like(point[0])

        self.function = f_free


class Harmonic(ModelPotential):
    """Generator for an n-dimensional harmonic potential.

    Passing arrays of omegas and minima to the constructor defines the
    attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        omega = kwargs.get('omega', np.full(dimension, 1))
        minimum = kwargs.get('minimum', np.full(dimension, 0))

        def f_harm(*coordinates):
            coordinates_length = np.size(coordinates, 0)
            if coordinates_length != self.dimension:
                raise ValueError('Input does not match potential dimension.')

            result = 0
            for i, coord in enumerate(coordinates):
                result += omega[i] * (coord-minimum[i]) ** 2
            return result

        self.function = f_harm


class Wall(ModelPotential):
    """Generator for an n-dimensional wall potential.

    Passing arrays of positions, widths, and heights to the constructor
    defines the attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', np.full(dimension, 1./2.))
        width = kwargs.get('width', np.full(dimension, 1./2.)) / 2.
        height = kwargs.get('height', 1./2.)

        def f_wall(*coordinates):

            coordinates_length = np.size(coordinates, 0)
            if coordinates_length != self.dimension:
                raise ValueError('Input does not match potential dimension.')

            result = np.full(np.shape(coordinates[0]), height)

            for i, coord in enumerate(coordinates):
                lower_bound = position[i]-width[i]
                upper_bound = position[i]+width[i]

                ar = np.piecewise(coord,
                                  [(coord >= lower_bound) &
                                   (coord < upper_bound)],
                                  [height])

                result = np.where(ar == 0, ar, result)

            return result

        self.function = f_wall


class Box(ModelPotential):
    """Generator for an n-dimensional box potential.

    Passing arrays of positions, widths, and heights to the constructor
    defines the attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', np.full(dimension, 1./2.))
        width = kwargs.get('width', np.full(dimension, 1./2.)) / 2.
        height = kwargs.get('height', 1000000)

        if isinstance(position,(list,np.ndarray)):
            pass
        else:
            position = np.array(position)
        if isinstance(width,(list,np.ndarray)):
            pass
        else:
            width = np.array(width)
        

        def f_box(*coordinates):

            coordinates_length = np.size(coordinates, 0)
            if coordinates_length != self.dimension:
                raise ValueError('Input does not match potential dimension.')

            result = np.full(np.shape(coordinates[0]), 0)

            for i, coord in enumerate(coordinates):
                upper_bound = position[i]+width[i]
                lower_bound = position[i]-width[i]

                ar = np.piecewise(coord,
                                  [(coord >= upper_bound) |
                                   (coord < lower_bound)],
                                  [height])

                result = np.where(ar == height, ar, result)

            return result

        self.function = f_box


class DoubleBox(ModelPotential):
    """Generator for an n-dimensional double box potential.

    Passing arrays of positions, widths, and heights to the constructor
    defines the attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position1 = kwargs.get('position1', np.full(dimension, 1./2.))
        position2 = kwargs.get('position2', np.full(dimension, 3./2.))
        width1 = kwargs.get('width1', np.full(dimension, 1./4.)) / 2.
        width2 = kwargs.get('width2', np.full(dimension, 1./4.)) / 2.
        height = kwargs.get('height', 1000000)

        def f_double_box(*coordinates):

            coordinates_length = np.size(coordinates, 0)
            if coordinates_length != self.dimension:
                raise ValueError('Input does not match potential dimension.')

            result = np.full(np.shape(coordinates[0]), 0)

            for i, coord in enumerate(coordinates):
                upper_bound = position1[i]+width1[i]
                lower_bound = position1[i]-width1[i]
                upper_bound2 = position2[i]+width2[i]
                lower_bound2 = position2[i]-width2[i]

                ar = np.piecewise(coord,
                                  [((coord >= upper_bound) |
                                   (coord < lower_bound)) &
                                   ((coord >= upper_bound2) |
                                   (coord < lower_bound2))],
                                  [height])

                result = np.where(ar == height, ar, result)

            return result

        self.function = f_double_box


class Morse(ModelPotential):
    """Generator for an n-dimensional morse potential.

    Passing arrays of a's, D's, and p's to the constructor
    defines the attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        morse_a = kwargs.get('morse_a', np.full(dimension, 0.5))
        morse_D = kwargs.get('morse_D', np.full(dimension, 5.0))
        morse_p = kwargs.get('morse_p', np.full(dimension, 0.5))

        def f_morse(*coordinates):
            coordinates_length = np.size(coordinates, 0)
            if coordinates_length != self.dimension:
                raise ValueError('Input does not match potential dimension.')

            result = 0.
            for i, coord in enumerate(coordinates):
                exponential = np.exp(-morse_a[i]*(coord-morse_p[i]))
                result += morse_D[i]*(1-exponential)**2

            return result

        self.function = f_morse


class DoubleSlit(ModelPotential):
    """Generator for an 2-dimensional double slit potential.

    Passing arrays of positions, widths, and heights to the constructor
    defines the attributes of the potential in each dimension.
    """

    def __init__(self, dimension, **kwargs):
        if dimension != 2:
            raise ValueError('Elbow only available in 2D.')
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', 0)
        thickness = kwargs.get('thickness', 1)
        spacing = kwargs.get('spacing', 1)
        height = kwargs.get('height', 100)
        centre = kwargs.get('centre', 0)
        slit_size = kwargs.get('slit_size', 0.8)

        def f_slit(x, y):
            lower = position - thickness*0.5
            upper = position + thickness*0.5

            slit_start_one = centre - spacing - slit_size
            slit_end_one = centre - spacing + slit_size
            slit_start_two = centre + spacing - slit_size
            slit_end_two = centre + spacing + slit_size

            result = x
            x_truth = np.logical_and(np.greater(x, lower), np.less(x, upper))
            lower_y = np.less(y, slit_start_one)
            upper_y = np.greater(y, slit_end_two)
            mid_y = np.logical_and(np.greater(y, slit_end_one),
                                   np.less(y, slit_start_two))

            all_y = lower_y + upper_y + mid_y
            all = np.logical_and(all_y, x_truth)

            result = np.where(all, height, 0)

            return result

        self.function = f_slit


class Elbow(ModelPotential):
    """Generator for an 2-dimensional elbow potential.

    Various keyword arguments can be given to specify the size and shape of the
    elbow.
    """

    def __init__(self, dimension, **kwargs):
        if dimension != 2:
            raise ValueError('Elbow only available in 2D.')
        ModelPotential.__init__(self, dimension)

        elbow_sc = kwargs.get('elbow_scale', 2.)
        elbow_p1 = kwargs.get('elbow_pos1', [11, 4.])
        elbow_p2 = kwargs.get('elbow_pos2', [4., 31./3.])
        elbow_si1 = kwargs.get('elbow_sigma1', [9./2., 1.])
        elbow_si2 = kwargs.get('elbow_sigma2', [3./2., 11./2.])

        def f_elbow(x, y):
            z2 = np.exp(-(1./2.)*(((x-y-0.1)/2.)**2 + ((x-y-0.1)/2.)**2))
            z = 100. * (
                -create_real_gaussian2D(x, y, x0=elbow_p1, sigma=elbow_si1)
                - create_real_gaussian2D(x, y, x0=elbow_p2, sigma=elbow_si2)
                ) + (50./3.)*z2
            return np.real(elbow_sc*z)

        self.function = f_elbow
