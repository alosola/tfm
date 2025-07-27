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

        
    def normalize(self):
        # # Normalize data by dividing by the "quiet sun" value at each wavelength
        # if hasattr(self, 'mean_quiet'):
        #     print("Normalizing", self.name, "data with quiet sun profile")
        #     self.data_n = self.data / self.mean_quiet[0]
        # else:
        #     print("Error normalizing data, quiet sun not set")

        # Normalizing data with value xnormalized = (x - xminimum) / range of x
        print("Normalizing data for", self.name, "and saving to object")
        self.data_n = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        

    def mean_quiet_region(self, xmin, xmax, ymin, ymax):
        # Calculate the mean intensity of the quiet region selected, for each wavelength
        print("Calculating quiet sun profile for object ", self.name)
        self.mean_quiet = self.data[ymin:ymax,xmin:xmax,:].mean(axis=(0,1))


    def calc_first_derivative(self):
        # Calculate first derivative for each point

        # Initialize empty array
        self.data_d = np.empty(self.data.shape)

        # Cycle through wavelengths
        self.data_d[:,:,0] = self._forwards_diff(0)
        self.data_d[:,:,self.size_wave - 1] = self._backwards_diff(self.size_wave - 1)
        for k in range(self.size_wave - 1):
            self.data_d[:,:,k] = self._central_diff(k)

    
    def _forwards_diff(self, wave_index):
        # Forward differentiation
        return (self.data_n[:,:,wave_index + 1] - self.data_n[:,:,wave_index]) / (self.wave_array[wave_index + 1] - self.wave_array[wave_index])
    

    def _backwards_diff(self, wave_index):
        # Backwards differentiation
        return (self.data_n[:,:,wave_index] - self.data_n[:,:,wave_index - 1]) / (self.wave_array[wave_index] - self.wave_array[wave_index - 1])
    

    def _central_diff(self, wave_index):
        # Central differentiation
        return (self.data_n[:,:,wave_index + 1] - self.data_n[:,:,wave_index - 1]) / (self.wave_array[wave_index + 1] - self.wave_array[wave_index - 1])


    def calc_second_derivative(self):
        # Calculate first derivative for each point

        # Initialize empty array
        self.data_dd = np.empty(self.data.shape)

        # Cycle through wavelengths
        self.data_dd[:,:,0] = (self.data_n[:,:,1] - self.data_n[:,:,0]) / (self.wave_array[1] - self.wave_array[0])
        self.data_dd[:,:,self.size_wave - 1] = (self.data_n[:,:,self.size_wave - 1] - self.data_n[:,:,self.size_wave - 2]) / (self.wave_array[self.size_wave - 1] - self.wave_array[self.size_wave - 2])
        for k in range(self.size_wave - 1):
            self.data_dd[:,:,k] = (self.data_d[:,:,k + 1] - self.data_d[:,:,k - 1]) / (self.wave_array[k + 1] - self.wave_array[k - 1])


    def plot_frame(self, wave_to_plot):
        # Plot non-normalized data
        return self._plot_frame(self.data, "", wave_to_plot)


    def plot_frame_n(self, wave_to_plot):
        # Plot normalized data
        return self._plot_frame(self.data_n, "", wave_to_plot)


    def plot_frame_d(self, wave_to_plot):
        # Plot derived data
        return self._plot_frame(self.data_d, "first derivative", wave_to_plot)
    

    def plot_frame_dd(self, wave_to_plot):
        # Plot second derived data
        return self._plot_frame(self.data_dd, "second derivative", wave_to_plot)


    def _plot_frame(self, data, title_tag, wave_to_plot):
        # Plot a single frame corresponding to a certain wavelength

        if hasattr(self, 'wave_array'):
            if (wave_to_plot < 112):
                # Assume you are selecting a frame index
                title = self.name + " data, " + title_tag + " for wavelength " + str(round(self.wave_array[wave_to_plot],3)) + " in frame " + str(wave_to_plot)
                fig, _ = plot_data(data[:,:,wave_to_plot], title)
            else:
                # Assume you are selecting the wavelength
                index = min(range(len(self.wave_array)), key=lambda i: abs(self.wave_array[i]-wave_to_plot))
                title = self.name + " data, " + title_tag + " for wavelength " + str(round(self.wave_array[index],3)) + " in frame " + str(index)
                fig, _ = plot_data(data[:,:,index], title)
        else:
            try:
                # If the wavelength hasn't been calibrated, you are forced to select only the index
                title = self.name + " data, " + title_tag + " for wavelength index " + str(wave_to_plot)
                fig, _ = plot_data(data[:,:,wave_to_plot], title)
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
        nrows = 14
        ncols = 8
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(50, 80), dpi=25) # TODO: fix padding here

        for i in range(nrows):
            for j in range(ncols):
                # print("Image #", str(i*ncols +j), "in spot ", str(i), ", ", str(j))
                ax[i,j].set_title(self.name + " data, index " + str(i*ncols +j))
                img = ax[i,j].imshow(self.data[:,:,(i*ncols + j)], cmap='gray')

        return fig