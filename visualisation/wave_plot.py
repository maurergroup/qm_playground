import numpy as np
from bokeh.models import Range1d, LinearAxis

from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.layouts import column
from bokeh.driving import repeat
from bokeh.palettes import Spectral8


class WavePlot:

    def __init__(self, data):

        self.raw_data = data

        self.x = self.raw_data['x']
        self.N = len(self.x)
        self.rho_t = self.raw_data['rho_t'].real
        self.nstates = int(len(self.rho_t[0]) / self.N)

        self.calculate_potential()

        self.setup_plot()

        func = self.get_update_function()
        curdoc().add_periodic_callback(func, 50)

    def calculate_potential(self):
        self.v_matrix = self.raw_data['V']
        if self.nstates == 1:
            self.V = np.diag(self.v_matrix)
        elif self.nstates == 2:
            N = self.N
            v1 = []
            v2 = []
            for i in range(N):
                mat = np.block([[self.v_matrix[i, i], self.v_matrix[i, i+N]],
                                [self.v_matrix[i+N, i], self.v_matrix[i+N, i+N]]])
                vals, vecs = np.linalg.eigh(mat)
                v1.append(vals[0])
                v2.append(vals[1])
            self.V = np.concatenate((v1, v2))

    def get_update_function(self):

        t = range(len(self.raw_data['rho_t']))

        @repeat(sequence=t)
        def update(i):
            self.update_plot(i)

        return update

    def update_plot(self, i):
        x = np.hstack((self.x, [np.max(self.x), np.min(self.x)]))

        rho = self.rho_t[i]
        rho_1 = rho[:self.N]
        y1 = np.hstack((rho_1, [0, 0]))
        self.sources[0].data = dict(x=x, y=y1)

        if self.nstates == 2:
            rho_2 = rho[self.N:]
            y2 = np.hstack((rho_2, [0, 0]))
            self.sources[1].data = dict(x=x, y=y2)

    def setup_plot(self):

        self.setup_energy_plot()
        self.setup_wave_movie()

    def setup_energy_plot(self):
        self.energy_plot = figure(plot_width=360, plot_height=180)
        self.energy_plot.toolbar_location = None

        y = self.raw_data['E_t']
        mean = np.mean(y)
        x = np.linspace(0, 1, len(y))
        self.energy_plot.line(x=x, y=(y-mean)/mean)

    def setup_wave_movie(self):
        self.wave_movie = figure(plot_width=720, plot_height=360)
        self.wave_movie.toolbar_location = None

        self.plot_potential()
        self.plot_waves()

    def plot_waves(self):
        min = np.min(self.rho_t)
        max = np.max(self.rho_t)
        range = max - min
        min -= 0.1 * range
        max += 0.1 * range

        self.wave_movie.y_range = Range1d(min, max)
        self.sources = [ColumnDataSource(data=dict(x=[], y=[])),
                        ColumnDataSource(data=dict(x=[], y=[]))]
        self.wave_movie.patch('x', 'y', source=self.sources[0],
                              color=Spectral8[0], alpha=0.7,
                              line_color='black')
        self.wave_movie.patch('x', 'y', source=self.sources[1],
                              color=Spectral8[-1], alpha=0.7,
                              line_color='black')

    def plot_potential(self):
        min = np.min(self.V)
        max = np.max(self.V)
        range = max - min
        min -= 0.1 * range
        max += 0.1 * range
        self.wave_movie.extra_y_ranges = {'pot_y': Range1d(min, max)}
        pot_axis = LinearAxis(y_range_name='pot_y')
        self.wave_movie.add_layout(pot_axis, 'right')

        self.plot_adiabatic_surfaces()
        self.plot_diabatic_elements()

    def plot_adiabatic_surfaces(self):
        v1 = ColumnDataSource(data=dict(x=self.x, y=self.V[:self.N]))
        self.wave_movie.line('x', 'y', source=v1, y_range_name='pot_y',
                             line_width=2, color=Spectral8[0], alpha=0.7)
        self.wave_movie.line('x', 'y', source=v1, y_range_name='pot_y',
                             line_width=2, line_dash='dashed', color='black')

        if self.nstates == 2:
            v2 = ColumnDataSource(data=dict(x=self.x, y=self.V[self.N:]))
            self.wave_movie.line('x', 'y', source=v2, y_range_name='pot_y',
                                 line_width=2, color=Spectral8[-1], alpha=0.7)
            self.wave_movie.line('x', 'y', source=v2, y_range_name='pot_y',
                                 line_width=2, line_dash='dashed', color='black')

    def plot_diabatic_elements(self):
        self.d11 = ColumnDataSource(data=dict(x=self.x,
                                    y=np.diag(self.v_matrix[:self.N, :self.N])))
        self.wave_movie.line('x', 'y', source=self.d11, y_range_name='pot_y',
                             line_dash='dashed', alpha=0.7,
                             color=Spectral8[0])

        self.d22 = ColumnDataSource(data=dict(x=self.x,
                                    y=np.diag(self.v_matrix[self.N:, self.N:])))
        self.wave_movie.line('x', 'y', source=self.d22, y_range_name='pot_y',
                             line_dash='dashed', alpha=0.7,
                             color=Spectral8[-1])

        self.d12 = ColumnDataSource(data=dict(x=self.x,
                                    y=np.diag(self.v_matrix[:self.N, self.N:])))
        self.wave_movie.line('x', 'y', source=self.d12, y_range_name='pot_y',
                             line_dash='dashed', alpha=0.7,
                             color='purple')

    def get_layout(self):
        return column(self.energy_plot, self.wave_movie)
