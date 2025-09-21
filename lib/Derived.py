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

# Global variable for line cuttoff
global line_cuttoff
line_cutoff = 55


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
        print("Calculating total polarization")
        global line_cutoff
        sum_n = 0
        sum_d = 0
        self.mp = np.zeros(np.shape(self.I.data[:,:,0:2]))

        # For first line
        for i in range(0,line_cutoff):
            sum_n += np.sqrt(self.Q.data_n[:,:,i]**2 + self.U.data_n[:,:,i]**2 + self.V.data_n[:,:,i]**2)
            sum_d += self.I.data_n[:,:,i]

        self.mp[:,:,0] = sum_n / sum_d

        # For second line
        for i in range(line_cutoff,self.I.data.shape[2]):
            sum_n += np.sqrt(self.Q.data_n[:,:,i]**2 + self.U.data_n[:,:,i]**2 + self.V.data_n[:,:,i]**2)
            sum_d += self.I.data_n[:,:,i]
        self.mp[:,:,1] = sum_n / sum_d


    def linear_polarization(self):
        # Calculate linear polarization of datacube (remove V, only Q and U)
        print("Calculating linear polarization")
        global line_cutoff
        sum_n = 0
        sum_d = 0
        self.lp = np.zeros(np.shape(self.I.data[:,:,0:2]))

        # For first line
        for i in range(0,line_cutoff):
            sum_n += np.sqrt(self.Q.data[:,:,i]**2 + self.U.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]

        self.lp[:,:,0] = sum_n / sum_d

        # For second line
        for i in range(line_cutoff,self.I.data.shape[2]):
            sum_n += np.sqrt(self.Q.data[:,:,i]**2 + self.U.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]
        self.lp[:,:,1] = sum_n / sum_d


    def circular_polarization(self):
        # Calculate circular polarization of datacube (remove Q and U, only V)
        print("Calculating circular polarization")
        global line_cutoff
        sum_n = 0
        sum_d = 0
        self.cp = np.zeros(np.shape(self.I.data[:,:,0:2]))

        # For first line
        for i in range(0,line_cutoff):
            sum_n += np.sqrt(self.V.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]

        self.cp[:,:,0] = sum_n / sum_d

        # For second line
        for i in range(line_cutoff,self.I.data.shape[2]):
            sum_n += np.sqrt(self.V.data[:,:,i]**2)
            sum_d += self.I.data[:,:,i]
        self.cp[:,:,1] = sum_n / sum_d


    def weak_field(self, Bv, Bt, chi, gamma):
        print(f'Saving weak field approximation to derived object')
        self.weak = WeakField()
        self.weak.Bv = Bv
        self.weak.Bt = Bt
        self.weak.chi = chi
        self.weak.gamma = gamma

    def strong_field(self, B, chi, gamma, binary_mask):
        print(f'Saving strong field approximation to derived object')
        self.strong = StrongField()
        self.strong.B = B
        self.strong.chi = chi
        self.strong.gamma = gamma
        self.binary_mask = binary_mask

    def combined_field(self, chi, gamma):
        print(f'Saving combined field approximation to derived object')
        self.B = B
        self.chi = chi
        self.gamma = gamma

    def save_pickle(self, filename='generated/objects/derived.pickle'):
        print("Saving datacube to pickle file:", filename)
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)