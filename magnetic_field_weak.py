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
from functions.sums import sw
from functions.load_pickles import load_pickles

print(f'Running script to calculate magnetic field using weak field approximation')

stokes_list, derived = load_pickles()

# Extract each Stokes parameter, to make it easier to work with
I = stokes_list['I']
Q = stokes_list['Q']
U = stokes_list['U']
V = stokes_list['V']

# Calculate first and second derivatives of Intensity data
I.calc_derivatives()



line_cuttoff = 55 # index at which we divide the data, between the spectral lines
f = 1 # filling factor, assumed to be 1




######### Calculate the vertical (longitudinal) component
print(f'Calculating longitudinal component of magnetic field')

lambda0  = [6301.51, 6302.50]  # in Angstroms
gbar     = [1.669, 2.487] # Lozitsky
C1 = [4.6686e-13 * l**2 * g for l, g in zip(lambda0, gbar)]
print(f"C1_1: {C1[0]}, C1_2: {C1[1]}")

Bv = [-sw((V.data_n * I.data_d)[:, :, :line_cuttoff]) / (C1[0] * f * sw((I.data_d ** 2)[:, :, :line_cuttoff])),
       -sw((V.data_n * I.data_d)[:, :, line_cuttoff:]) / (C1[1] * f * sw((I.data_d ** 2)[:, :, line_cuttoff:]))]


######### Calculate the horizontal (transverse) component
print(f'Calculating transverse component of magnetic field')

Gbar = [g**2 for g in gbar]  # Landi Degl'Innocenti & Landolfi (2004), si la línea es un triplete #TODO
C2 = [5.4490e-26 * l**4 * g for l, g in zip(lambda0, Gbar)]
print(f"C2_1: {C2[0]}, C2_2: {C2[1]}")

L = np.sqrt(Q.data_n**2 + U.data_n**2)
Bt = [sw(L[:,:,:line_cuttoff] * np.abs(I.data_dd[:,:,:line_cuttoff])) / (C2[0] * f * sw(np.abs(I.data_dd[:,:,:line_cuttoff]))**2),
       sw(L[:,:,line_cuttoff:] * np.abs(I.data_dd[:,:,line_cuttoff:])) / (C2[1] * f * sw(np.abs(I.data_dd[:,:,line_cuttoff:]))**2)]
Bt = np.sqrt(Bt)


######### Compute the azimuth angle (equal for both lines, does not depend on wavelength or Landé factor)
print(f'Calculating azimuth angle of magnetic field')

chi = np.arctan2(sw(U.data_n * I.data_dd), sw(Q.data_n * I.data_dd)) / 2


######### Calculate inclination angle
print(f'Calculating inclintion angle of magnetic field')

const = [4/3 * g**2/G for g, G in (gbar, Gbar)]

num = [sw(np.abs(I.wave_array[:line_cuttoff] - lambda0[0]) * np.abs(L[:,:,:line_cuttoff]) * V.data_n[:,:,:line_cuttoff]**2 * np.abs(I.data_d[:,:,:line_cuttoff])),
         sw(np.abs(I.wave_array[line_cuttoff:] - lambda0[1]) * np.abs(L[:,:,line_cuttoff:]) * V.data_n[:,:,line_cuttoff:]**2 * np.abs(I.data_d[:,:,line_cuttoff:]))]
denom = [sw(np.abs(V.data_n[:,:,:line_cuttoff])**4),
         sw(np.abs(V.data_n[:,:,line_cuttoff:])**4)]
tan2gamma = [const[0] * num[0] / denom[0],
             const[1] * num[1] / denom[1]]
gamma = np.arctan(np.sqrt(tan2gamma))




## Save weak field approximation to object
derived.weak_field(Bv, Bt, chi, gamma)

# Save updated derived object
derived.save_pickle()