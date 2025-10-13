
import pickle

# Open data files
from functions.load_pickles import load_pickles
stokes_list, _ = load_pickles(select="stokes")
I = stokes_list['I']

# Calculate derivatives
I.calc_derivatives()

# Save updated Stokes objects
stokes_filename = "generated/objects/stokes.pickle"
print("Saving datacube to pickle file:", stokes_filename)
with open(stokes_filename, 'wb') as handle:
    pickle.dump(stokes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)