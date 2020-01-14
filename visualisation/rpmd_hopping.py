import itertools

import colorcet
import numpy as np
from bokeh.plotting import figure

from rpmd_plot1D import RPMDPlot1D


class RPMDHoppingPlot1D(RPMDPlot1D):

    # def setup_plot(self):
    #     self.particle_movie = figure(toolbar_location=None)

    #     self.plot_potential()
    #     self.plot_particles()

    def plot_potential(self):
        self.calculate_potential()
        self.x = np.linspace(self.cell[0][0], self.cell[0][1], self.N)
        self.particle_movie.line(x=self.x, y=self.v1)
        self.particle_movie.line(x=self.x, y=self.v2)

    def calculate_potential(self):
        v11 = self.raw_data['v11']
        v12 = self.raw_data['v12']
        v22 = self.raw_data['v22']
        self.N = len(v11)
        v1 = []
        v2 = []
        for i in range(self.N):
            mat = np.block([[v11[i], v12[i]],
                            [v12[i], v22[i]]])
            vals, vecs = np.linalg.eigh(mat)
            v1.append(vals[0])
            v2.append(vals[1])
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)

    def update_plot(self, i):
        r_t = self.raw_data['r_t']
        state_t = self.raw_data['state_occ_t']

        x = r_t[i, :].flatten()
        indices = np.digitize(x, self.x)
        state = state_t[i, :].flatten()

        ys = []
        for bead in range(len(indices)):
            n = state[bead]
            if n == 0:
                ys.append(self.v1[indices[bead]])
            elif n == 1:
                ys.append(self.v2[indices[bead]])

        self.source.data = dict(x=x, y=ys)

    def plot_energy(self):
        colors = itertools.cycle(colorcet.glasbey)

        E_t = self.raw_data['E_t']
        E_kin_t = self.raw_data['E_pot_t']
        E_pot_t = self.raw_data['E_kin_t']
        E_kink_t = self.raw_data['E_kink_t']
        print(E_kink_t.shape)
        energies = [E_t, E_kin_t, E_pot_t, E_kink_t]
        nparticles = np.shape(E_t)[1]
        x = np.linspace(0, 1, len(E_t))
        for i in range(nparticles):
            for e, color in zip(energies, colors):
                self.energy_plot.line(x=x, y=(e[:, i]), color=color)
