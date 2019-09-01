import numpy as np
from bokeh.models import Range1d, LinearAxis
from surface3d import Surface3d

from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import repeat
from bokeh.palettes import Spectral8


class WavePlot2D:

    def __init__(self, data):

        self.raw_data = data

        # self.x = self.raw_data['x']
        # self.N = len(self.x)
        # self.rho_t = self.raw_data['rho_t'].real
        self.psi_t = self.raw_data['psi_t'].real
        # self.nstates = int(len(self.rho_t[0]) / self.N)

        self.setup_plot()

        func = self.get_update_function()
        curdoc().add_periodic_callback(func, 50)

    def get_update_function(self):

        t = range(len(self.raw_data['psi_t']))

        @repeat(sequence=t)
        def update(i):
            self.update_plot(i)

        return update

    def update_plot(self, i):

        size = self.raw_data['N']
        cell = self.raw_data['cell']
        x = np.linspace(cell[0][0], cell[1][0], size)
        y = np.linspace(cell[0][1], cell[1][1], size)
        xx, yy = np.meshgrid(x, y)
        z = self.psi_t[i]
        z = z.reshape((size, size))
        self.source.data = dict(x=xx, y=yy, z=z)

    def setup_plot(self):

        self.setup_wave_movie()

    def setup_wave_movie(self):
        self.source = ColumnDataSource(data=dict(x=[], y=[]))
        self.wave_movie = Surface3d(x='x', y='y', z='z',
                                    data_source=self.source)

    # def plot_waves(self):
    #     # min = np.min(self.rho_t)
    #     # max = np.max(self.rho_t)
    #     # range = max - min
    #     # min -= 0.1 * range
    #     # max += 0.1 * range

    #     # self.wave_movie.y_range = Range1d(min, max)

    def get_layout(self):
        return column(self.wave_movie)
