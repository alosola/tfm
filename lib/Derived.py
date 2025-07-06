# Define Derived parameters class

# TODO
# This class will contain all the data related to a Stokes parameter:
# - original datacube (X, Y, wavelength axes)
# - calibrated wavelenth array
# - normalized datacube
# - quiet sun profile

# as well as functions to plot figures


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from functions.plot_data import plot_data


class Derived:
    def __init__(self, stokes_list):
        self.I = stokes_list['I']
        self.Q = stokes_list['Q']
        self.U = stokes_list['U']
        self.V = stokes_list['V']


    def mean_polarization(self):
        # Calculate mean polarization of datacube
        self.mp = np.empty(self.I.data.shape)
        for i in range(112):
            self._mean_polarization_wave(i)


    def _mean_polarization_wave(self, i):
        # Calculate the mean polarization in a given frame
        self.mp[:,:,i] = np.sqrt(self.Q.data[:,:,i]**2 + self.U.data[:,:,i]**2 + self.V.data[:,:,i]**2) / self.I.data[:,:,i]

    
    def plot_mean_polarization(self, i):
        plot_data(self.mp[:,:,i], "Polarization data, for wavelength index " + str(i))