#cell 0
from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import copy
import pickle

# Project modules
from lib.Stokes import Stokes
from lib.MidpointNormalize import MidpointNormalize
from lib.MidPointNorm import MidPointNorm
from functions.plot_data import plot_data
from functions.plot_angle_gradient import plot_angle_gradient
from functions.sw import sw
from functions.noise_level import noise_level

#cell 1
# Open data files
from functions.load_pickles import load_pickles
stokes_list, derived = load_pickles(select="stokes")
T_cont = pickle.load(open('generated/objects/T_cont.pickle', "rb"))

#cell 2
# Extract each Stokes parameter into dictionary, to make it easier to work with
I = stokes_list['I']
Q = stokes_list['Q']
U = stokes_list['U']
V = stokes_list['V']

PLOT_FIGURES=0

#NOTE: these values are divided into _1 for the first spectral line and _2 for the second spectral line
lambda0  = [6301.5008, 6302.4932]  # in Angstroms
gbar     = [1.669, 2.487] # Lozitsky
C1 = [4.6686e-13 * l**2 * g for l, g in zip(lambda0, gbar)]
Gbar = [g**2 for g in gbar]  # Landi Degl'Innocenti & Landolfi (2004), si la línea es un triplete #TODO
C2 = [5.4490e-26 * l**4 * g for l, g in zip(lambda0, Gbar)]

# Strong field approximation
C = 4.67e-13 # TODO: where is this constant from? Sara's email (and PDF) 29/08

kB = 1.3806488e-16 # [erg K-1]
h = 6.6260755e-27  # [erg s]
c = 2.99792458e10  # [cm · s−1]
Teff = 5780 # [K] T quiet sun average
f = 1 # filling factor, assumed to be 1

M = 55.845 # Fe atomic mass, [g mol-1]
av = 6.02214076e23 # avogadro, [mol-1]
m =  M/av
Xi = 0 # microturbulence, assumed 0

Icont = I.data_n[:,:,:5].mean(axis=2) # I map in continuum
line_cuttoff = 55 # index at which we divide the data, between the spectral lines

# Calculate the vertical (longitudinal) component
Bv = [-sw((V.data_n * I.data_d)[:, :, :line_cuttoff]) / (C1[0] * f * sw((I.data_d ** 2)[:, :, :line_cuttoff])),
       -sw((V.data_n * I.data_d)[:, :, line_cuttoff:]) / (C1[1] * f * sw((I.data_d ** 2)[:, :, line_cuttoff:]))]


print(f'Max Bv [G]:\nFirst line: {np.max(Bv[0])}\nSecond line: {np.max(Bv[1])}')

if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_data(Bv[i], colourmap='berlin_r', norm=MidPointNorm(0), colourbar_label=r'$B_{||}$ [G]')
        fig.savefig("generated/" + f"WFA_Bv_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/WFA_Bv_{i}.png")


# Calculate the horizontal (transverse) component
L = np.sqrt(Q.data_n**2 + U.data_n**2)
Bt = [sw(L[:,:,:line_cuttoff] * np.abs(I.data_dd[:,:,:line_cuttoff])) / (C2[0] * f * sw(np.abs(I.data_dd[:,:,:line_cuttoff]))**2),
       sw(L[:,:,line_cuttoff:] * np.abs(I.data_dd[:,:,line_cuttoff:])) / (C2[1] * f * sw(np.abs(I.data_dd[:,:,line_cuttoff:]))**2)]
Bt = np.sqrt(Bt)

print(f'Max Bt [G]:\nFirst line: {np.max(Bt[0])}\nSecond line: {np.max(Bt[1])}')


if PLOT_FIGURES:
    for i in range(2):
        if i==0:
            divnorm=MidpointNormalize(vmin=50, vmax=500, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=50, vmax=600, midpoint=0)

        fig, _, _ = plot_data(Bt[i], colourmap='berlin_r', norm=divnorm, colourbar_label=r'$B_{||}$ [G]')
        fig.savefig("generated/" + f"WFA_Bt_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/WFA_Bt_{i}.png")

B_WFA = [np.sqrt(Bv[0]**2 + Bt[0]**2),
         np.sqrt(Bv[1]**2 + Bt[1]**2)]
print(f'Max B [G]:\nFirst line: {np.max(B_WFA[0])}\nSecond line: {np.max(B_WFA[1])}')


if PLOT_FIGURES:
    for i in range(2):
        if i==0:
            divnorm=MidpointNormalize(vmin=50, vmax=4500, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=50, vmax=3000, midpoint=0)

        fig, _, _ = plot_data(B_WFA[i], colourmap='berlin_r', norm=divnorm, colourbar_label=r'$B$ [G]')
        fig.savefig("generated/" + f"WFA_B_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/WFA_B_{i}.png")

# Compute the inclination angle, simpler
theta = [np.arctan(Bt[0] / Bv[0]),
         np.arctan(Bt[1] / Bv[1])]

# Set all negative values of theta to theta + pi
for theta_inst in theta:
    theta_inst[theta_inst < 0] += np.pi


if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_angle_gradient(theta[i], colourmap='PRGn_r', colourbar_label=r'$\theta$ [deg]')
        fig.savefig("generated/" + f"WFA_theta_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/WFA_theta_{i}.png")


# Compute the azimuth angle (equal for both lines, does not depend on wavelength or Landé factor)
shape = np.shape(I.data_n[:,:,0])
num = [sw(U.data_n[:, :, :line_cuttoff] * I.data_dd[:, :, :line_cuttoff]),
       sw(U.data_n[:, :, line_cuttoff:] * I.data_dd[:, :, line_cuttoff:])]
den = [sw(Q.data_n[:, :, :line_cuttoff] * I.data_dd[:, :, :line_cuttoff]),
       sw(Q.data_n[:, :, line_cuttoff:] * I.data_dd[:, :, line_cuttoff:])]
phi = [0.5 * np.arctan(num[0]/den[0]),
       0.5 * np.arctan(num[1]/den[1])]

for k in range(2):
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if den[k][i,j] < 0:
                phi[k][i,j] += np.pi/2
            elif num[k][i,j] < 0 and den[k][i,j] > 0:
                phi[k][i,j] += 2*np.pi
            if den[k][i,j] == 0:
                if num[k][i,j] > 0:
                    phi[k][i,j] += np.pi/4
                elif num[k][i,j] < 0:
                    phi[k][i,j] += 3/4*np.pi


    if PLOT_FIGURES:
        fig, _, _ = plot_angle_gradient(phi[k], colourbar_label=r'$\phi$ [deg]', scale=[0,180])
        fig.savefig("generated/" + f"WFA_phi_{k}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/WFA_phi_{k}.png")


# # Calculate delta lambda_B for each pixel, for each line
if (False):
    dlB = np.zeros(I.data.shape)[:,:,:2].astype(float)
    dlB_binary = np.zeros(I.data.shape)[:,:,:2].astype(float)
    margin = 3.5
    verbose = 0

    for i in range(0, np.shape(V.data_n)[0]):

        # Read out every 50 rows
        if np.mod(i,50) == 0:
            print(f'Row {i} of {np.shape(V.data_n)[0]}')

        for j in range(0, np.shape(V.data_n)[1]):
            if (verbose):
                print(f'For pixel {i},{j}:')

            # Initialize variables
            peaks_p = [0, 0]
            peaks_n = [0, 0]

            # Get the spectrum for this pixel
            spectrum = copy.copy(V.data_n[i,j,:])

            # Calculate noise level for spectrum region outside the spectral lines
            sd = noise_level(spectrum[90:])
            if (verbose):
                print(f'Noise level = {sd}')

            # First line
            peaks_p[0] = int(np.argmax(spectrum[:60])) if spectrum[:60].max() > margin*sd else None
            peaks_n[0] = int(np.argmin(spectrum[:60])) if spectrum[:60].min() < -margin*sd else None

            # # Second line
            peaks_p[1] = 60 + int(np.argmax(spectrum[60:])) if spectrum[60:].max() > margin*sd else None
            peaks_n[1] = 60 + int(np.argmin(spectrum[60:])) if spectrum[60:].min() < -margin*sd else None

            if (verbose):
                print(f'Peak positions: {peaks_p}, {peaks_n}')

            # If two peaks have been found for the first line, calculate dlB
            if peaks_p[0] is not None and peaks_n[0] is not None:
                dlB_binary[i, j, 0] = 1
                dlB[i, j, 0] = np.array(V.wave_array)[peaks_p[0]] - np.array(V.wave_array)[peaks_n[0]]

                if (verbose):
                    plt.figure()
                    plt.vlines(V.wave_array[peaks_n[0]], ymin=np.min(spectrum), ymax=np.max(spectrum), color='g', linestyle='--')
                    plt.vlines(V.wave_array[peaks_p[0]], ymin=np.min(spectrum), ymax=np.max(spectrum), color='purple', linestyle='--')

                    print(f'Peak 1 positive: {V.wave_array[peaks_p[0]]}')
                    print(f'Peak 1 negative: {V.wave_array[peaks_n[0]]}')
                    print(np.array(V.wave_array)[peaks_p[0]] - np.array(V.wave_array)[peaks_n[0]])
                    print(f'Distance: {dlB[i, j, 0]:.6f} Angstrom')
            else:
                dlB[i, j, 0] = np.nan


            # If two peaks have been found for the second line, calculate dlB
            if peaks_p[1] is not None and peaks_n[1] is not None:
                dlB_binary[i, j, 1] = 1
                dlB[i, j, 1] = np.array(V.wave_array)[peaks_p[1]] - np.array(V.wave_array)[peaks_n[1]]

                if (verbose):
                    plt.figure()
                    plt.vlines(V.wave_array[peaks_n[1]], ymin=np.min(spectrum), ymax=np.max(spectrum), color='g', linestyle='--')
                    plt.vlines(V.wave_array[peaks_p[1]], ymin=np.min(spectrum), ymax=np.max(spectrum), color='purple', linestyle='--')

                    print(f'Peak 2 positive: {V.wave_array[peaks_p[1]]}')
                    print(f'Peak 2 negative: {V.wave_array[peaks_n[1]]}')
                    print(np.array(V.wave_array)[peaks_p[1]] - np.array(V.wave_array)[peaks_n[1]])
                    print(f'Distance: {dlB[i, j, 1]:.6f} Angstrom')
            else:
                dlB[i, j, 1] = np.nan

            if (verbose):
                plt.hlines(margin*sd, V.wave_array[0], V.wave_array[-1], color='red', linestyle='--')
                plt.hlines(-margin*sd, V.wave_array[0], V.wave_array[-1], color='red', linestyle='--')
                plt.plot(V.wave_array, spectrum, linestyle='-', color='black')

    with open('generated/objects/dlB.pickle', 'wb') as f:
        pickle.dump(dlB, f)
else:
    dlB = -pickle.load(open('generated/objects/dlB.pickle', "rb"))


# Select pixels of interest
verbose=1
plate_scale_x = 0.14857 # arcseconds per pixel
plate_scale_y = 0.16 # arcseconds per pixel
x_pix = np.array([500, 180, 290, 400, 320, 300])
y_pix = np.array([118, 113, 538, 354, 428, 413])
pix_name=['A', 'B', 'C', 'D', 'E', 'F']
margin = 3

fig, axs = plt.subplots(3, 2, figsize=(9, 7.5))

for i in range(6):

    # Initialize variables
    peaks_p = [0, 0]
    peaks_n = [0, 0]

    # Get the spectrum for this pixel
    spectrum = copy.copy(V.data_n[y_pix[i],x_pix[i],:])

    # Calculate noise level for spectrum region outside the spectral lines
    sd = noise_level(spectrum[90:])

    # First line
    peaks_p[0] = int(np.argmax(spectrum[:60])) if spectrum[:60].max() > margin*sd else None
    peaks_n[0] = int(np.argmin(spectrum[:60])) if spectrum[:60].min() < -margin*sd else None

    # # Second line
    peaks_p[1] = 60 + int(np.argmax(spectrum[60:])) if spectrum[60:].max() > margin*sd else None
    peaks_n[1] = 60 + int(np.argmin(spectrum[60:])) if spectrum[60:].min() < -margin*sd else None

    if (verbose):
        row, col = divmod(i, 2)
        ax = axs[row, col]
        ax.set_title(f'Pixel {pix_name[i]}: '+r'$\Delta \lambda_B$' f'= [{dlB[y_pix[i], x_pix[i], 0]:.4f}, {dlB[y_pix[i], x_pix[i], 1]:.4f}] ' + r'$\AA$')

        # Only plot if there are no None values in peaks_p or peaks_n
        if all(p is not None for p in peaks_p) and all(n is not None for n in peaks_n):
            for k in range(2):
                ax.axvline(V.wave_array[peaks_p[k]], color='purple', linewidth=0.8)
                ax.axvline(V.wave_array[peaks_n[k]], color='green', linewidth=0.8)

        ax.plot(I.wave_array, V.data_n[y_pix[i],x_pix[i],:], label='I data', linewidth=0.4, marker='.', markersize=5)
        # Remove inner axes
        if row < 2 and col < 2:
            ax.set_xticks([])
        else:
            ax.set_xlabel(r'Wavelength [$\AA$]')

        ax.set_ylabel(r'$ V\:/\:I_{0,QS}$')


if PLOT_FIGURES:
    axs[0, 0].set_title('Pixel A (peak SNR insufficient)')
    plt.tight_layout()

    fig.savefig("generated/" + f"delta_lambda_b_pixels.png", dpi=200, bbox_inches='tight')
    print("Saved figure to file", f"generated/delta_lambda_b_pixels.png")


# Doppler broadening of the spectral line
# Rebecca Centeno, equation 3
dlD = np.array([lambda0[0]/c * np.sqrt(2*kB*T_cont/m + Xi**2),
                lambda0[1]/c * np.sqrt(2*kB*T_cont/m + Xi**2)]) # will be in units of lambda, in this case Angstrom


if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_data(dlD[0], colourmap='managua', norm=None, colourbar_label=r'$\Delta \lambda _D$ [$\AA$]')
        fig.savefig("generated/" + f"delta_lambda_d_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/delta_lambda_d_{i}.png")

    divnorm=MidpointNormalize(vmin=0.-0.3, vmax=0.3, midpoint=0)
    for i in range(2):
        fig, _, _ = plot_data(dlB[:,:,i], colourmap='managua', norm=divnorm, colourbar_label=r'$\Delta \lambda _B$ [$\AA$]')
        fig.savefig("generated/" + f"delta_lambda_b_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/delta_lambda_b_{i}.png")


# Remove outliers
dlB[dlB > 0.3] = np.nan
dlB[dlB < -0.3] = np.nan


theta_SFA = [np.arctan((sw(Q.data_n[:,:,:line_cuttoff])**2 + sw(U.data_n[:,:,:line_cuttoff])**2)**0.5/sw(V.data_n[:,:,:line_cuttoff])),
             np.arctan(sw(Q.data_n[:,:,line_cuttoff:]**2 + U.data_n[:,:,line_cuttoff:]**2)**0.25/sw(V.data_n[:,:,line_cuttoff:]))]


# Set all negative values of theta to theta + pi
for theta_inst in theta_SFA:
    theta_inst[theta_inst < 0] += np.pi


if True:
    for i in range(2):
        fig, _, _ = plot_angle_gradient(theta_SFA[i], colourmap='PRGn_r', colourbar_label=r'$\theta$ [deg]')
        fig.savefig("generated/" + f"SFA_theta_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_theta_{i}.png")


phi_SFA = [0.5 * np.arctan(sw(U.data_n[:,:,:line_cuttoff]) / sw(Q.data_n[:,:,:line_cuttoff])),
           0.5 * np.arctan(sw(U.data_n[:,:,line_cuttoff:]) / sw(Q.data_n[:,:,line_cuttoff:]))]

if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_angle_gradient(phi_SFA[i], colourbar_label=r'$\phi$ [deg]')
        fig.savefig("generated/" + f"SFA_phi_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_phi_{k}.png")


B_SFA = [dlB[:,:,0] / (2 * C * np.array(lambda0[0])**2 * np.array(gbar[0])),
         dlB[:,:,1] / (2 * C * np.array(lambda0[1])**2 * np.array(gbar[1]))]


if PLOT_FIGURES:
    for i in range(2):
        if i==0:
            divnorm=MidpointNormalize(vmin=-4000, vmax=4000, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=-3000, vmax=3000, midpoint=0)

        fig, _, _ = plot_data(B_SFA[i], colourmap='berlin_r', norm=divnorm, colourbar_label=r'$B$ [G]')
        fig.savefig("generated/" + f"SFA_B_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_B_{i}.png")


Bv_SFA = [B_SFA[0] * np.cos(theta_SFA[0]),
          B_SFA[1] * np.cos(theta_SFA[1])]


Bt_SFA = [B_SFA[0] * np.sin(theta_SFA[0]),
          B_SFA[1] * np.sin(theta_SFA[1])]


if True:
    for i in range(2):
        if i==0:
            divnorm=MidpointNormalize(vmin=-4000, vmax=4000, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=-3000, vmax=3000, midpoint=0)

        fig, _, _ = plot_data(Bt_SFA[i], colourmap='berlin_r', norm=None, colourbar_label=r'$B_{||}$ [G]')
        fig.savefig("generated/" + f"SFA_Bt_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_Bt_{i}.png")

        if i==0:
            divnorm=MidpointNormalize(vmin=50, vmax=4500, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=50, vmax=3000, midpoint=0)

        fig, _, _ = plot_data(Bv_SFA[i], colourmap='berlin_r', norm=None, colourbar_label=r'$B$ [G]')
        fig.savefig("generated/" + f"WFA_Bv_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_Bv_{i}.png")


# #cell 37
# Bv = np.moveaxis(Bv, 0, -1)
# Bt = np.moveaxis(Bt, 0, -1)

# #cell 38
# # Constants
# kB = 1.3806488e-16 # [erg K-1]
# h = 6.6260755e-27  # [erg s]
# c = 2.99792458e10  # [cm · s−1]
# lambda0  = np.array([6301.51*1e-8, 6302.50*1e-8])   # Angstroms to cm
# l = np.mean(lambda0)
# Icont = I.data_n[:,:,:5].mean(axis=2) # I map in continuum
# Teff = 5780 # [K] T quiet sun average
# T = (1/Teff - kB*l/(h*c) * np.log(Icont))**-1
# M = 55.845 # Fe atomic mass, [g mol-1]
# av = 6.022e23 # avogadro, [mol-1]
# m =  M/av
# Xi = 0 # microturbulence, assumed 0


# dlD = np.moveaxis(dlD, 0, -1)
# np.shape(dlD)


# #cell 39
# # remove outliers
# B_strong[B_strong > 5000] = np.nan
# B_strong[B_strong < -5000] = np.nan

# #cell 40
# B = np.empty(Bv.shape)
# B[:,:,:] = np.nan
# B_mask = np.zeros(Bv.shape) * np.nan
# # gamma = np.zeros(derived.weak.Bv.shape) * np.nan
# # chi = np.zeros(derived.weak.Bv.shape) * np.nan

# for i in range(I.data.shape[0]):
#     for j in range(I.data.shape[1]):
#         for k in range(0,2):
#             if (dlB_binary[i,j,k] == 0) or np.isnan(dlB[i,j,k]) or (np.abs(dlB[i,j,k]) < np.abs(dlD[i,j,k])):
#                 # If dlB could not be calculated, or if it is less than the doppler effect
#                 B[i,j,k] = np.sqrt(Bv[i,j,k]**2 + Bt[i,j,k]**2)
#                 B_mask[i,j,k] = 0
#                 # chi[i,j,k] = derived.weak.chi[i,j,k]
#                 # gamma[i,j,k] = derived.weak.gamma[i,j,k]
#             else:
#                 # Use strong field approximation otherwise
#                 B[i,j,k] = B_strong[i,j,k]
#                 B_mask[i,j,k] = 1
#                 # chi[i,j,k] = derived.strong.chi[i,j,k]
#                 # gamma[i,j,k] = derived.strong.gamma[i,j,k]

# #cell 41
# print(np.abs(dlD[300,300,0]))
# print(np.abs(dlB[300,300,0]))

# #cell 42
# dlB_clean = np.copy(dlB[:,:,0])
# dlB_clean[dlB_clean > 0.5] = np.nan
# dlB_clean[dlB_clean < -0.5] = np.nan

# #cell 43
# plot_data(B[:,:,0], colourmap='bwr', title=r"B combined [G]")

# #cell 44

# divnorm=MidpointNormalize(vmin=-4000, vmax=4000, midpoint=0)
# plot_data(B_strong, colourmap='berlin_r', title=r"B SFA [G]")
# plot_data(np.abs(B_strong)*np.sin(gamma), norm=divnorm, colourmap='berlin_r', title=r"Bt SFA [G]")
# plot_data(np.abs(B_strong)*np.cos(gamma), norm=divnorm, colourmap='berlin_r', title=r"Bv SFA [G]")

# #cell 45
# theta_SFA = 0.5 * np.arctan(sw.)

# #cell 46
# plot_data(dlB[:,:,0] > dlD[:,:,0], colourmap='grey')

# #cell 47
# # Plot Gaussian fits for a few example pixels
# import matplotlib.pyplot as plt

# example_pixels = [
#     (300, 300),
#     (118, 500),
#     (113, 180),
#     (538, 290),
#     (354, 400),
#     (428, 320)
# ]

# for (i, j) in example_pixels:
#     spectrum = copy.copy(V.data_n[i, j, :])
#     x_full = V.wave_array
#     fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#     for line_idx, (start, end, label) in enumerate([(0, 60, 'Line 1'), (60, 112, 'Line 2')]):
#         x = x_full[start:end]
#         y = spectrum[start:end]
#         axs[line_idx].plot(x, y, 'k.', label='Data')
#         # Fit positive lobe
#         if y.max() > margin * noise_level(spectrum[90:]):
#             idx_p = np.argmax(y)
#             window = 5
#             fit_slice = slice(max(0, idx_p - window), min(len(x), idx_p + window + 1))
#             try:
#                 from scipy.optimize import curve_fit
#                 popt, _ = curve_fit(gaussian, x[fit_slice], y[fit_slice],
#                                     p0=[y[idx_p], x[idx_p], 0.1, np.median(y)])
#                 axs[line_idx].plot(x[fit_slice], gaussian(x[fit_slice], *popt), 'r-', label='Gaussian fit (pos)')
#                 axs[line_idx].axvline(popt[1], color='r', linestyle='--', label='Fit peak (pos)')
#             except Exception:
#                 axs[line_idx].axvline(x[idx_p], color='r', linestyle='--', label='Max (pos)')
#         # Fit negative lobe
#         if y.min() < -margin * noise_level(spectrum[90:]):
#             idx_n = np.argmin(y)
#             window = 5
#             fit_slice = slice(max(0, idx_n - window), min(len(x), idx_n + window + 1))
#             try:
#                 popt, _ = curve_fit(gaussian, x[fit_slice], y[fit_slice],
#                                     p0=[y[idx_n], x[idx_n], 0.1, np.median(y)])
#                 axs[line_idx].plot(x[fit_slice], gaussian(x[fit_slice], *popt), 'b-', label='Gaussian fit (neg)')
#                 axs[line_idx].axvline(popt[1], color='b', linestyle='--', label='Fit peak (neg)')
#             except Exception:
#                 axs[line_idx].axvline(x[idx_n], color='b', linestyle='--', label='Min (neg)')
#         axs[line_idx].set_title(f'Pixel ({i},{j}) {label}')
#         axs[line_idx].set_xlabel('Wavelength [Å]')
#         axs[line_idx].set_ylabel('V')
#         axs[line_idx].legend()
#     plt.tight_layout()
#     plt.show()

