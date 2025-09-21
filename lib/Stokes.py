# Define Stokes parameter class

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


class Stokes:
    def __init__(self, name, data):
        self.name = name
        self.data = np.flipud(data.swapaxes(0,1))

        self.size_x = self.data.shape[0]
        self.size_y = self.data.shape[1]
        self.size_wave = self.data.shape[2]


    def add_wavelength(self, wave_array):
        # Add array of calibrated wavelength, indicating to which wavelength (in nm)
        # each index in the 3rd dimension of the datacube corresponds
        print("Saving wavelength calibration to", self.name, "object")
        self.wave_array = wave_array


    def normalize(self, I_cont):
        # Normalizing data by dividing by the "quiet sun" Intensity continuum value
        print(f"Normalizing data for {self.name} with I continuum value {I_cont} and saving to object")
        self.data_n = self.data / I_cont


    def mean_quiet_region(self, xmin, xmax, ymin, ymax):
        # Calculate the mean intensity of the quiet region selected, for each wavelength
        print("Calculating quiet sun profile for object ", self.name)
        self.mean_quiet = self.data[ymin:ymax,xmin:xmax,:].mean(axis=(0,1))


    def calc_derivatives(self):
        # Calculate first derivative for each wavelength
        print(f'Calculating first derivative of {self.name} data')
        self.data_d = np.gradient(self.data_n, self.wave_array.ravel(), axis=2)

        # Calculate second derivative for each wavelength
        print(f'Calculating second derivative of {self.name} data')
        self.data_dd = np.gradient(self.data_d, self.wave_array.ravel(), axis=2)


    def plot_frame(self, wave_to_plot):
        # Plot non-normalized data
        return self._plot_frame(self.data, "", wave_to_plot, colourbar_label='Number of counts')


    def plot_frame_n(self, wave_to_plot):
        # Plot normalized data
        return self._plot_frame(self.data_n, "", wave_to_plot)


    def plot_frame_d(self, wave_to_plot):
        # Plot derived data
        return self._plot_frame(self.data_d, "first derivative", wave_to_plot)


    def plot_frame_dd(self, wave_to_plot):
        # Plot second derived data
        return self._plot_frame(self.data_dd, "second derivative", wave_to_plot)


    def _plot_frame(self, data, title_tag, wave_to_plot, colourbar_label=None):
        # Plot a single frame corresponding to a certain wavelength

        if hasattr(self, 'wave_array'):
            if (wave_to_plot < 112):
                # Assume you are selecting a frame index
                title = self.name + " data, " + title_tag + " for wavelength " + str(round(self.wave_array[wave_to_plot],3)) + " in frame " + str(wave_to_plot)
                fig, _ = plot_data(data[:,:,wave_to_plot], title, colourbar_label=colourbar_label)
            else:
                # Assume you are selecting the wavelength
                index = min(range(len(self.wave_array)), key=lambda i: abs(self.wave_array[i]-wave_to_plot))
                title = self.name + " data, " + title_tag + " for wavelength " + str(round(self.wave_array[index],3)) + " in frame " + str(index)
                fig, _ = plot_data(data[:,:,index], title, colourbar_label=colourbar_label)
        else:
            try:
                # If the wavelength hasn't been calibrated, you are forced to select only the index
                title = self.name + " data, " + title_tag + " for wavelength index " + str(wave_to_plot)
                fig, _ = plot_data(data[:,:,wave_to_plot], title, colourbar_label=colourbar_label)
            except:
                print("Wrong index specified, or wavelength not calibrated")

        return fig


    def plot_all_frames(self):
        # Plot non-normalized data, all frames
        return self._plot_all_frames(self.data)


    def plot_all_frames_n(self):
        # Plot normalized data, all frames
        return self._plot_all_frames(self.data_n)


    def _plot_all_frames(self, data):
        # Plot all frames of data, corresponding to all wavelenths
        nrows = 8
        ncols = 14
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 7), dpi=25)
        max_count = np.max(data)
        min_count = np.min(data)

        for i in range(nrows):
            for j in range(ncols):
                # print("Image #", str(i*ncols +j), "in spot ", str(i), ", ", str(j))
                # ax[i,j].set_title(self.name + " data, index " + str(i*ncols +j))
                img = ax[i,j].imshow(self.data[:,:,(i*ncols + j)], cmap='gray', vmin=min_count, vmax=max_count)
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)
                ax[i,j].spines.values

        fig.tight_layout()

        return fig