from bokeh.models import TextInput, Button
from bokeh.plotting import curdoc
from bokeh.layouts import column, grid
from wave_plot import WavePlot
from traj_plot2D import TrajPlot2D
from traj_plot1D import TrajPlot1D
import pickle

"""
Run with 'bokeh server --show plot_server.py'
"""


class FileReader():

    def __init__(self):

        self.file_input = TextInput(placeholder='Enter file path')
        self.layout = grid([], ncols=2)
        self.i = 0
        self.j = 0

    def add_plot_area(self):
        curdoc().add_root(self.layout)

    def construct_plot(self):
        self.load_data()

        if self.data.mode == 'wave':
            self.plot = WavePlot(self.data)

        elif self.data.mode == 'traj':
            ndim = len(self.data.potential.shape)
            if ndim == 1:
                self.plot = TrajPlot1D(self.data)
            elif ndim == 2:
                self.plot = TrajPlot2D(self.data)

        elif self.data.mode == 'rpmd':
            self.plot = TrajPlot2D(self.data)

        self.layout.children.append((self.plot.get_layout(), self.i, self.j))
        if self.j == 1:
            self.i += 1
            self.j = 0
        else:
            self.j += 1

    def load_data(self):
        file = open(self.file_input.value, 'rb')
        self.data = pickle.load(file)


reader = FileReader()

button = Button(label="Plot data!")
button.on_click(reader.construct_plot)

format = column(reader.file_input, button)
curdoc().add_root(format)
reader.add_plot_area()
