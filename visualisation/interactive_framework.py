"""
This is very bad but I'm not going to delete it just in case someone can later
learn from my mistakes. It could be used as a reasonable starting point in the
future but you have to be careful not to let things get too tangled up.
"""

from qmp import Model
from qmp.integrator.waveintegrators import SOFT_Propagator
from qmp.systems.grid import Grid1D
from qmp.potential import Potential
from qmp.potential import preset_potentials
from qmp.integrator.dyn_tools import create_gaussian
import numpy as np

from bokeh.layouts import column, grid, row
from bokeh.models import Button, FileInput, Select, ColumnDataSource
from bokeh.models import TextInput
from bokeh.models.widgets.buttons import Toggle, Dropdown
from bokeh.models.widgets.sliders import Slider, RangeSlider
from bokeh.models.widgets.groups import RadioButtonGroup
from bokeh.plotting import figure, curdoc
import pickle
import base64


file = FileInput()


def decode_data(data):
    decode = base64.b64decode(data)
    return pickle.loads(decode)


def load_data():
    data = decode_data(file.value)

    x = data['x']
    y = data['psi_t'][0, 0].real

    p = create_plot(x, y)

    curdoc().add_root(column(p))


def create_plot(x, y):
    # p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
    p = figure()
    p.line(x=x, y=y)
    return p


class ModelSettings:

    def __init__(self):

        self.mode_select = Select(title='Select the calculation type:',
                             options=[('wave', 'Wavepacket'),
                                      ('rpmd', 'Ring Polymer Molecular Dynamics'),
                                      ('hop', 'Surface Hopping'),
                                      ('traj', 'Classical Trajectory')],
                             value='wave',
                             )
        self.mode_select.on_change("value", self.update_mode)

        self.dimension_choice = RadioButtonGroup(labels=['1D', '2D'], active=0)
        self.dimension_choice.on_click(self.update_dimension)

        self.cell_slider = RangeSlider(title='Resize the cell', start=-25,
                                       end=25, step=1, value=(-10, 10))
        self.cell_slider.on_change("value", self.update_cell)

        self.length_slider = Slider(title='Choose the number of steps',
                                    start=0, end=1e4,
                                    step=1e2, value=5e3)
        self.length_slider.on_change("value", self.update_steps)

        self.dt_slider = Slider(title='Choose the timestep',
                                    start=0, end=100,
                                    step=1, value=1)
        self.dt_slider.on_change("value", self.update_dt)

        self.column = column(self.mode_select, self.dimension_choice,
                             self.cell_slider, self.length_slider)

        self.read_defaults()

        self.panel = row(self.column, self.mode_settings.get_column())

    def update_dimension(self, index):
        self.ndim = index + 1

    def update_mode(self, attr, old, new):
        self.mode = new
        if self.mode == 'wave':
            self.mode_settings = WavePacketSettings(self)

    def update_cell(self, attr, old, new):
        self.cell = [list(new)]

    def update_steps(self, attr, old, new):
        self.steps = new

    def update_dt(self, attr, old, new):
        self.dt = self.dt_slider.value

    def read_defaults(self):
        self.update_dimension(0)
        self.update_cell(0, 0, self.cell_slider.value)
        self.update_steps(0, 0, self.length_slider.value)
        self.update_mode(0, 0, self.mode_select.value)

    def get_panel(self):
        return self.panel


class PotentialPanel:

    def __init__(self, settings):
        self.settings = settings

        self.figure = figure()
        self.source = ColumnDataSource(data=dict(x=[0],
                                       y=[0]))
        self.figure.line('x', 'y', source=self.source)

        self.choose = Select(title='Choose the potential :',
                             options=['Free', 'Harmonic', 'Wall', 'Box'],
                             value='Harmonic',
                             )
        self.choose.on_change('value', self.update_potential)
        self.settings.cell_slider.on_change('value', self.update_potential)
        self.update_potential(0, 0, 0)

        self.column = column(self.choose, self.figure)

    def update_potential(self, attr, old, new):
        self.f = eval('preset_potentials.'
                      + self.choose.value
                      + f'({self.settings.ndim})')
        self.potential = Potential(self.settings.cell, f=self.f())
        x = np.linspace(self.settings.cell[0][0],
                        self.settings.cell[0][1], 100)
        y = self.potential(x)
        self.source.data = dict(x=x, y=y)

    def get_column(self):
        return self.column


class WavePacketSettings:

    def __init__(self, settings):
        self.settings = settings

        self.mass_choose = TextInput(title='Mass: ',
                                     value='2000')
        self.r_choose = TextInput(title='Position: ',
                                  value='0')
        self.p_choose = TextInput(title='Momentum: ',
                                  value='1')
        self.N_choose = TextInput(title='Grid Size: ',
                                  value='256')
        for w in [self.mass_choose, self.r_choose,
                  self.p_choose, self.N_choose]:
            w.on_change('value', self.update_settings)

        self.settings.cell_slider.on_change('value', self.update_settings)

        self.column = column(self.mass_choose, self.r_choose, self.p_choose,
                             self.N_choose)

        self.update_settings(0, 0, 0)

    def update_settings(self, attr, old, new):
        self.mass = float(self.mass_choose.value)
        self.r = float(self.r_choose.value)
        self.p = float(self.p_choose.value)
        self.N = int(self.N_choose.value)
        self.x = np.linspace(self.settings.cell[0][0],
                             self.settings.cell[0][1], self.N)

        self.system = Grid1D(self.mass,
                             self.settings.cell[0][0],
                             self.settings.cell[0][1],
                             self.N)

        self.create_initial_psi()

    def create_initial_psi(self):
        sigma = 0.5
        self.psi = create_gaussian(self.x, x0=self.r, p0=self.p, sigma=sigma)
        self.psi /= np.sqrt(np.conjugate(self.psi).dot(self.psi))
        self.system.set_initial_wvfn(self.psi)

    def get_column(self):
        return self.column


def run_calculation():
    pass


settings = ModelSettings()

pot = PotentialPanel(settings)

run_button = Button(label="Run Calculation")
run_button.on_click(run_calculation)

curdoc().add_root(settings.get_panel())
curdoc().add_root(pot.get_column())
curdoc().add_root(run_button)
