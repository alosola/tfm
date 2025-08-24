import pickle
from pathlib import Path


def load_pickles(stokes_filename="generated/objects/stokes.pickle", derived_filename="generated/objects/derived.pickle"):
    # Check that the Stokes object file exists
    my_file = Path(stokes_filename)
    if not my_file.is_file():
        print("Stokes object does not exist (", stokes_filename,"), please run file init_data.py first")
        stokes = None
    else:
        # Load the object from the file
        print("Loading Stokes data from file", stokes_filename)
        stokes_list = pickle.load(open(stokes_filename, "rb"))

    # Check that the derived object file exists
    my_file = Path(derived_filename)
    if not my_file.is_file():
        print("Derived object does not exist (", derived_filename,"), please run file init_data.py first")
        derived = None
    else:
        # Load the object from the file
        print("Loading Derived data from file", derived_filename)
        derived = pickle.load(open(derived_filename, "rb"))

    return stokes_list, derived