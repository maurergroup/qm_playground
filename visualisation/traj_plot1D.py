from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.driving import repeat
from bokeh.layouts import row
import numpy as np


class TrajPlot1D:

    def __init__(self, data):

        self.raw_data = data

        self.cell = self.raw_data['cell']

        self.setup_plot()

        func = self.get_update_function()
        curdoc().add_periodic_callback(func, 50)

    def setup_plot(self):
        self.particle_movie = figure(toolbar_location=None)

        self.plot_potential()
        self.plot_particles()

    def plot_potential(self):
        pot = self.raw_data['potential']
        x = np.linspace(self.cell[0][0], self.cell[0][1], len(pot))
        self.particle_movie.line(x=x, y=pot)

    def plot_particles(self):

        self.source = ColumnDataSource(data=dict(x=[], y=[]))
        self.particle_movie.circle('x', 'y', size=10, fill_color='white',
                                   line_color='black', line_width=3,
                                   source=self.source)

    def get_update_function(self):

        t = range(len(self.raw_data['r_t']))

        @repeat(sequence=t)
        def update(i):
            self.update_plot(i)

        return update

    def update_plot(self, i):
        r_t = self.raw_data['r_t']
        x = r_t[i, :]
        y = np.zeros_like(x)
        self.source.data = dict(x=x, y=y)

    def get_layout(self):
        return row(self.particle_movie)
