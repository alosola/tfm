## Calculate magnetic field


from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    print("Stokes object does not exist (", stokes_filename,"), please run file init_data.py first")

# Check that the Nova object file exists
nova_filename = "generated/objects/nova.pickle"
my_file = Path(nova_filename)
if not my_file.is_file():
    print("Nova object does not exist (", nova_filename,"), please run file init_data.py first")

# Load the objects from the files
print("Loading Stokes data from file", stokes_filename)
stokes_list = pickle.load(open(stokes_filename, "rb"))
print("Loading Nova data from file", nova_filename)
nova = pickle.load(open(nova_filename, "rb"))


# Calculate first derivative
stokes_list['I'].calc_first_derivative()

# Calculate second derivative
stokes_list['I'].calc_second_derivative()