## Calculate polarization


from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import copy
import pickle

# Project modules
from lib.Stokes import Stokes
from lib.Derived import Derived
from functions.plot_data import plot_data


# Check that the Stokes object file exists
stokes_filename = "generated/objects/stokes.pickle"
my_file = Path(stokes_filename)
if not my_file.is_file():
    print("Stokes object does not exists (", stokes_filename,"), please run file init_data.py first")

# Load the object from the file
print("Loading Stokes data from file", stokes_filename)
stokes_list = pickle.load(open(stokes_filename, "rb"))

# Initializing derived parameter object
nova = Derived(stokes_list)

# Calculate total polarization
print("Calculaing total polarization")
nova.total_polarization() # the reference for this is currently Ana's slide about HMI - find an article
# fig, _ = nova.plot_total_polarization(scale=[0, 1])

# Calculate linear polarization
print("Calculaing linear polarization")
nova.linear_polarization()
# fig, _ = nova.plot_linear_polarization()

# Calculate circular polarization
print("Calculaing circular polarization")
nova.circular_polarization()
# fig, _ = nova.plot_circular_polarization()

# Save updated objects
derived_filename = "generated/objects/derived.pickle"
print("Saving datacube to pickle file:", derived_filename)
with open(derived_filename, 'wb') as handle:
    pickle.dump(nova, handle, protocol=pickle.HIGHEST_PROTOCOL)