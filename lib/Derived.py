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
import pickle


class WeakField:
    def __init__(self):
        self.Bv = None
        self.Bt = None
        self.chi = None
        self.gamma = None

class StrongField:
    def __init__(self):
        self.B = None
        self.chi = None
        self.gamma = None

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
            sum_n += np.sqrt(self.Q.data[:,:,i]**2 + self.U.data[:,:,i]**2 + self.V.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]
        self.mp = sum_n / sum_d


    def linear_polarization(self):
        # Calculate linear polarization of datacube (remove V)
        sum_n = 0
        sum_d = 0
        for i in range(112):
            sum_n += np.sqrt(self.Q.data[:,:,i]**2 + self.U.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]
        self.lp = sum_n / sum_d


    def circular_polarization(self):
        # Calculate circular polarization of datacube (remove Q and U)
        sum_n = 0
        sum_d = 0
        for i in range(112):
            sum_n += np.sqrt(self.V.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]
        self.cp = sum_n / sum_d

    def weak_field(self, Bv, Bt, chi, gamma):
        print(f'Saving weak field approximation to derived object')
        self.weak = WeakField()
        self.weak.Bv = Bv
        self.weak.Bt = Bt
        self.weak.chi = chi
        self.weak.gamma = gamma

    def strong_field(self, B, chi, gamma):
        print(f'Saving strong field approximation to derived object')
        self.strong = StrongField()
        self.strong.B = B
        self.strong.chi = chi
        self.strong.gamma = gamma

    def plot_total_polarization(self, scale=None):
        return plot_data(self.mp, "Total polarization", scale=scale)

    def plot_linear_polarization(self, scale=None):
        return plot_data(self.lp, "Linear polarization", scale=scale)

    def plot_circular_polarization(self, scale=None):
        return plot_data(self.cp, "Circular polarization", scale=scale)

    def save_pickle(self, filename='generated/objects/derived.pickle'):
        print("Saving datacube to pickle file:", filename)
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)