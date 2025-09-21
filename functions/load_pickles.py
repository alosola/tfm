import pickle
from pathlib import Path


def load_pickles(stokes_filename="generated/objects/stokes.pickle", derived_filename="generated/objects/derived.pickle", select="both"):
    # Check that the Stokes object file exists
    my_file = Path(stokes_filename)
    stokes_list = None
    if not my_file.is_file():
        print("Stokes object does not exist (", stokes_filename,"), please run file init_data.py first")
    elif select=="stokes" or select=="both":
        # Load the object from the file
        print("Loading Stokes data from file", stokes_filename)
        stokes_list = pickle.load(open(stokes_filename, "rb"))

    # Check that the derived object file exists
    my_file = Path(derived_filename)
    derived = None
    if not my_file.is_file():
        print("Derived object does not exist (", derived_filename,"), please run file init_data.py first")
        derived = None
    elif select=="derived" or select=="both":
        # Load the object from the file
        print("Loading Derived data from file", derived_filename)
        derived = pickle.load(open(derived_filename, "rb"))

    return stokes_list, derived