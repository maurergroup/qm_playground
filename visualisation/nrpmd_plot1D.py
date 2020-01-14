import numpy as np
from rpmd_plot1D import RPMDPlot1D


class NRPMDPlot1D(RPMDPlot1D):

    def plot_potential(self):
        v11 = self.raw_data['v11']
        v12 = self.raw_data['v12']
        v22 = self.raw_data['v22']
        x = np.linspace(self.cell[0][0], self.cell[0][1], len(v11))
        self.particle_movie.line(x=x, y=v11)
        self.particle_movie.line(x=x, y=v12)
        self.particle_movie.line(x=x, y=v22)
