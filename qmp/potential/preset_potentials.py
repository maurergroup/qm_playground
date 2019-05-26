import numpy as np


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
            return 0.

        self.function = f_free


class Harmonic(ModelPotential):

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        omega = kwargs.get('omega', np.full(dimension, 1./2.))
        minimum = kwargs.get('minimum', np.full(dimension, 1./2.))

        def f_harm(*point):
            try:
                assert len(point) == self.dimension
            except AssertionError:
                print("Dimension of potential does not match input.")

            result = omega * (point-minimum)**2
            return np.sum(result)

        self.function = f_harm


class Wall(ModelPotential):

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', np.full(dimension, 1./2.))
        width = kwargs.get('width', np.full(dimension, 1./2.)) / 2.
        height = kwargs.get('height', 1./2.)

        def f_wall(*point):
            try:
                assert len(point) == self.dimension
            except AssertionError:
                print("Dimension of potential does not match input.")

            if (point >= position-width).all() and \
                    (point < position+width).all():
                result = height
            else:
                result = 0.

            return result

        self.function = f_wall


class Box(ModelPotential):

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        position = kwargs.get('position', np.full(dimension, 1./2.))
        width = kwargs.get('width', np.full(dimension, 1./2.)) / 2.
        height = kwargs.get('height', 1000000)

        def f_box(*point):
            try:
                assert len(point) == self.dimension
            except AssertionError:
                print("Dimension of potential does not match input.")

            if (point > position-width).all() and \
                    (point <= position+width).all():
                result = 0.
            else:
                result = height

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

        def f_double_box(*point):
            try:
                assert len(point) == self.dimension
            except AssertionError:
                print("Dimension of potential does not match input.")

            if (point > position1-width1).all() and \
                    (point <= position1+width1).all():
                result = 0.
            elif (point > position2-width2).all() and \
                    (point <= position2+width2).all():
                result = 0.
            else:
                result = height

            return result

        self.function = f_double_box


class Morse(ModelPotential):

    def __init__(self, dimension, **kwargs):
        ModelPotential.__init__(self, dimension)

        morse_a = kwargs.get('morse_a', np.full(dimension, 0.5))
        morse_D = kwargs.get('morse_D', np.full(dimension, 5.0))
        morse_p = kwargs.get('morse_pos', np.full(dimension, 0.5))

        def f_morse(*point):
            result = 0.
            try:
                assert len(point) == self.dimension
            except AssertionError:
                print("Dimension of potential does not match input.")

            exponential = np.exp(-morse_a*(point-morse_p))
            result = morse_D*(1-exponential)**2

            return np.sum(result)

        self.function = f_morse
