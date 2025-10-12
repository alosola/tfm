#cell 0
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import copy
import pickle
from astropy.visualization import make_lupton_rgb
from functions.plot_data import plot_data

# Project modules
from functions.plot_data import plot_data
from lib.Stokes import Stokes
from lib.Derived import Derived


# Open data files
from functions.load_pickles import load_pickles
_, derived = load_pickles(select='derived')


titles = False

for i in range(2):
    if titles:
        print("Plotting with titles")

        fig, _, _ = plot_data(derived.mp[:,:,i], f'Total polarization (630{i+1}.5 Å)', colourmap='gist_earth')
        fig.savefig("generated/" + f"3.total_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.total_polarization_{i}.png")

        fig, _, _ = plot_data(derived.lp[:,:,i], f'Linear polarization (630{i+1}.5 Å)', colourmap='gist_earth', size='small')
        fig.savefig("generated/" + f"3.linear_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.linear_polarization_{i}.png")

        fig, _, _ = plot_data(derived.cp[:,:,i], f'Circular polarization (630{i+1}.5 Å)', colourmap='gist_earth', size='small')
        fig.savefig("generated/" + f"3.circular_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.circular_polarization_{i}.png")
    else:
        print("Plotting without titles")

        fig, _, _ = plot_data(derived.mp[:,:,i], colourmap='gist_earth')
        fig.savefig("generated/" + f"3.total_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.total_polarization_{i}.png")

        fig, _, _ = plot_data(derived.lp[:,:,i], colourmap='gist_earth', size='small')
        fig.savefig("generated/" + f"3.linear_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.linear_polarization_{i}.png")

        fig, _, _ = plot_data(derived.cp[:,:,i], colourmap='gist_earth', size='small')
        fig.savefig("generated/" + f"3.circular_polarization_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/3.circular_polarization_{i}.png")
