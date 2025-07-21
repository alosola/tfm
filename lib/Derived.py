# Define Derived parameters class

# TODO
# This class will contain all the data related to a Stokes parameter:
# - original datacube (X, Y, wavelength axes)
# - calibrated wavelenth array
# - normalized datacube
# - quiet sun profile

# as well as functions to plot figures

import sys
sys.path.append('/home/solaa/workspace/tfm')

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


    def total_polarization(self):
        # Calculate total polarization of datacube
        sum_n = 0
        sum_d = 0
        for i in range(112):
            sum_n += np.sqrt(self.Q.data_normalized[:,:,i]**2 + self.U.data_normalized[:,:,i]**2 + self.V.data_normalized[:,:,i]**2)
            sum_d += self.I.data_normalized[:,:,i]
        self.mp = sum_n / sum_d


    def linear_polarization(self):
        # Calculate linear polarization of datacube (remove V)
        sum_n = 0
        sum_d = 0
        for i in range(112):
            sum_n += np.sqrt(self.Q.data_normalized[:,:,i]**2 + self.U.data_normalized[:,:,i]**2)
            sum_d += self.I.data_normalized[:,:,i]
        self.lp = sum_n / sum_d


    def circular_polarization(self):
        # Calculate circular polarization of datacube (remove Q and U)
        sum_n = 0
        sum_d = 0
        for i in range(112):
            sum_n += np.sqrt(self.V.data_normalized[:,:,i]**2)
            sum_d += self.I.data_normalized[:,:,i]
        self.cp = sum_n / sum_d

    
    def plot_total_polarization(self):
        return plot_data(self.mp, "Total polarization")

    def plot_linear_polarization(self):
        return plot_data(self.lp, "Linear polarization")

    def plot_circular_polarization(self):
        return plot_data(self.cp, "Circular polarization")