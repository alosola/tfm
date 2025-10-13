# Calculate the noise level of a signal using standard deviation
import numpy as np

def noise_level(a):
    a = np.asanyarray(a)
    sd = a.std(ddof=1)
    return sd