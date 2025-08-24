import numpy as np


# Define function to sum data along the third axis (sum all wavelengths)
def sw(data):
    return np.sum(data, axis=2)