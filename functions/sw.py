# Define function to sum data along the third axis (sum all wavelengths)
import numpy as np

def sw(data):
    return np.sum(data, axis=2)