## Initial data processing:
# - open h5py datafile
# - create Stokes parameter objects and populate
# - calculate normalized data
# - save object to file for loading from another file

import h5py
import pickle
from pathlib import Path

# Project modules
from lib.Stokes import Stokes


# Open datafile
datafile = 'data/AR_12665_133153_0.h5'
print("Reading data file", datafile)
data = h5py.File(datafile, 'r')
print("Opened data file", data.file)

# Print data parameters
print("List of keys in file:", list(data.keys()))
key = list(data.keys())[0]
print("    Number of strokes parameters:   ", data[key].shape[0])
print("    Size of X axis:                 ", data[key].shape[1])
print("    Size of Y axis:                 ", data[key].shape[2])
print("    Number of measured wavelengths: ", data[key].shape[3])

# Extract each Stokes parameter into dictionary
i = 0
I = Stokes('I', data['stokes'][0])
Q = Stokes('Q', data['stokes'][1])
U = Stokes('U', data['stokes'][2])
V = Stokes('V', data['stokes'][3])
stokes_list = {'I': I, 'Q': Q, 'U': U, 'V': V}
for stokes in stokes_list:
    print(stokes, 'shape:', stokes_list[stokes].data.shape)
    i = i + 1

# Save Stokes objects
Path("generated/objects/").mkdir(parents=True, exist_ok=True)
stokes_filename = "generated/objects/stokes.pickle"
print("Saving datacube to pickle file:", stokes_filename)
with open(stokes_filename, 'wb') as handle:
    pickle.dump(stokes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)