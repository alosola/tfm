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
    AQ = np.zeros(I.data.shape)[:,:,:2].astype(float)
    AU = np.zeros(I.data.shape)[:,:,:2].astype(float)
    AV = np.zeros(I.data.shape)[:,:,:2].astype(float)
    margin = 4
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
                plt.figure()

            # If two peaks have been found for the first line, calculate dlB
            for k in range(2):
                if peaks_p[k] is not None and peaks_n[k] is not None:
                    dlB_binary[i, j, k] = 1
                    dlB[i, j, k] = np.array(V.wave_array)[peaks_p[k]] - np.array(V.wave_array)[peaks_n[k]]
                    midpoint_index = int((peaks_p[k] + peaks_n[k]) / 2)
                    AV[i, j, k] = np.abs(spectrum[peaks_p[k]])
                    AQ[i, j, k] = np.abs(Q.data_n[i, j, midpoint_index] - np.mean([Q.data_n[i, j, peaks_n[k]], Q.data_n[i, j, peaks_p[k]]]))
                    AU[i, j, k] = np.abs(U.data_n[i, j, midpoint_index] - np.mean([U.data_n[i, j, peaks_n[k]], U.data_n[i, j, peaks_p[k]]]))

                    if (verbose):
                        plt.title(f'Pixel {i},{j}, line {k+1}, dlB = {dlB[i, j, k]:.4f} Å, AQ = {AQ[i, j, k]:.4f}, AU = {AU[i, j, k]:.4f}')
                        # plt.axvline(V.wave_array[peaks_n[k]], color='g', linestyle='--')
                        # plt.axvline(V.wave_array[peaks_p[k]], color='g', linestyle='--')
                        # plt.axvline(V.wave_array[midpoint_index], color='purple', linestyle='--')
                        # plt.axhline(np.min(U.data_n[i, j, midpoint_index-1:midpoint_index+1]), color='red', linestyle='--')
                        # plt.axhline(np.min(U.data_n[i, j, midpoint_index-1:midpoint_index+1])+AU[i, j, k], color='red', linestyle='--')
                        plt.axhline(np.min(Q.data_n[i, j, midpoint_index-1:midpoint_index+1]), color='red', linestyle='--')
                        plt.axhline(np.min(Q.data_n[i, j, midpoint_index-1:midpoint_index+1])+AQ[i, j, k], color='red', linestyle='--')
                        # plt.axhline(AU[i, j, k], color='green', linestyle='--')
                else:
                    dlB[i, j, k] = np.nan


            if (verbose):
                # plt.hlines(margin*sd, V.wave_array[0], V.wave_array[-1], color='red', linestyle='--')
                # plt.hlines(-margin*sd, V.wave_array[0], V.wave_array[-1], color='red', linestyle='--')
                plt.plot(V.wave_array, spectrum, linestyle='-', color='black')
                # plt.plot(U.wave_array, U.data_n[i, j, :], linestyle='-', color='grey')
                plt.plot(Q.wave_array, Q.data_n[i, j, :], linestyle='-', color='grey')

    with open('generated/objects/dlB.pickle', 'wb') as f:
        pickle.dump(dlB, f)
    with open('generated/objects/AQ.pickle', 'wb') as f:
        pickle.dump(AQ, f)
    with open('generated/objects/AU.pickle', 'wb') as f:
        pickle.dump(AU, f)
    with open('generated/objects/AV.pickle', 'wb') as f:
        pickle.dump(AV, f)
else:
    dlB = pickle.load(open('generated/objects/dlB.pickle', "rb"))
    AQ = pickle.load(open('generated/objects/AQ.pickle', "rb"))
    AU = pickle.load(open('generated/objects/AU.pickle', "rb"))
    AV = pickle.load(open('generated/objects/AV.pickle', "rb"))

dlB = -dlB # Make dlB positive

if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_data(AQ[:,:,i], colourbar_label=r'$A_Q$ [$\AA$]')
        fig.savefig("generated/" + f"AQ_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/AQ_{i}.png")

        fig, _, _ = plot_data(AU[:,:,i], colourbar_label=r'$A_U$ [$\AA$]')
        fig.savefig("generated/" + f"AU_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/AU_{i}.png")

        fig, _, _ = plot_data(AV[:,:,i], colourbar_label=r'$A_V$ [$\AA$]')
        fig.savefig("generated/" + f"AV_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/AV_{i}.png")

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


theta_SFA = [np.arctan((AQ[:,:,0]**2+AU[:,:,0]**2)**0.25/AV[:,:,0]),
             np.arctan((AQ[:,:,1]**2+AU[:,:,1]**2)**0.25/AV[:,:,1])]



# Set sign of theta from sign of dlB
for k in range(2):
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if dlB[i,j,k] < 0:
                theta_SFA[k][i,j] = np.pi - theta_SFA[k][i,j]
            if AV[i,j,k] == 0:
                theta_SFA[k][i,j] = np.nan


if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_angle_gradient(theta_SFA[i], colourmap='PRGn_r', colourbar_label=r'$\theta$ [deg]')
        fig.savefig("generated/" + f"SFA_theta_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_theta_{i}.png")



phi_SFA = [0.5 * np.arctan(AU[:,:,0] / AQ[:,:,0]),
           0.5 * np.arctan(AU[:,:,1] / AQ[:,:,1])]

if PLOT_FIGURES:
    for i in range(2):
        fig, _, _ = plot_angle_gradient(phi_SFA[i], colourbar_label=r'$\phi$ [deg]')
        fig.savefig("generated/" + f"SFA_phi_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_phi_{i}.png")


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


if PLOT_FIGURES:
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
        fig.savefig("generated/" + f"SFA_Bv_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/SFA_Bv_{i}.png")



# Combine the results from both methods, depending on the criteria for each pixel
B_combine = np.abs(B_SFA)
theta_combine = np.copy(theta_SFA)
phi_combine = np.copy(phi_SFA)

for k in range(2):
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if np.isnan(dlB[i,j,k]):
                phi_combine[k][i,j] = phi[k][i,j]
                theta_combine[k][i,j] = theta[k][i,j]
                B_combine[k][i,j] = B_WFA[k][i,j]




if True:
    for i in range(2):
        if i==0:
            divnorm=MidpointNormalize(vmin=50, vmax=4500, midpoint=0)
        elif i==1:
            divnorm=MidpointNormalize(vmin=50, vmax=3000, midpoint=0)

        fig, _, _ = plot_data(B_combine[i], colourmap='berlin_r', norm=divnorm, colourbar_label=r'$B$ [G]')
        fig.savefig("generated/" + f"B_combine_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/B_combine_{i}.png")



        fig, _, _ = plot_angle_gradient(theta_combine[i], colourmap='PRGn_r', colourbar_label=r'$\theta$ [deg]')
        fig.savefig("generated/" + f"theta_combine_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/theta_combine_{i}.png")


        fig, _, _ = plot_angle_gradient(phi_combine[i], colourbar_label=r'$\phi$ [deg]')
        fig.savefig("generated/" + f"phi_combine_{i}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/phi_combine_{i}.png")