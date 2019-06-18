import numpy as np
from qmp.integrator.dyn_tools import create_real_gaussian2D


class ModelPotential(object):

    def __init__(self, dimension):
        self.function = None
        self.dimension = dimension

    def __call__(self):
        return self.function


class Free(ModelPotential):

    def __init__(self, dimension):
        ModelPotential.__init__(self, dimension)

        def f_free(*point):
            return np.zeros_like(point[0])

        self.function = f_free


class Harmonic(ModelPotential):

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

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', np.full(dimension, 1./2.))
        width = kwargs.get('width', np.full(dimension, 1./2.)) / 2.
        height = kwargs.get('height', 1000000)

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


class Elbow(ModelPotential):

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
