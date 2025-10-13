from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from astropy.modeling import models, fitting
import h5py
import copy
import pickle

# Project modules
from lib.Stokes import Stokes
from lib.MidPointNorm import MidPointNorm
from functions.plot_data import plot_data
from functions.plot_angle_gradient import plot_angle_gradient

# Open data files
from functions.load_pickles import load_pickles
stokes_list, _ = load_pickles(select="stokes")

# Extract each Stokes parameter into dictionary, to make it easier to work with
I = stokes_list['I']
Q = stokes_list['Q']
U = stokes_list['U']
V = stokes_list['V']


#cell 4
lambda0  = [I.wave_array[29], I.wave_array[75]]   # Angstroms to cm
print(f"Reference wavelength: {lambda0[0]} Angstroms")

shape = np.shape(I.data_n[:,:,:2])

# Fit a Gaussian to the data and find its minimum
verbose = 0

shape = np.shape(I.data_n[:,:,:2])
fit_quality = np.zeros(shape)
min_x_array = np.zeros(shape)
dl = np.zeros(shape)
line_cuttoff = 56

recalculate=0
if (recalculate):
    for k in range(2):
        # for i in range(200,212):
        for i in range(0, shape[0]):
            if np.mod(i,10) == 0:
                print(f'Row {i} of {np.shape(I.data_n)[0]}')
            # for j in range(200,212):
            for j in range(0, shape[1]):
                if k==0:
                    x = I.wave_array[:line_cuttoff]
                    y = I.data_n[i,j,:line_cuttoff]
                else:
                    x = I.wave_array[line_cuttoff:]
                    y = I.data_n[i,j,line_cuttoff:]

                # Normalize y for fitting stability
                y_norm = (y - np.median(y)) / np.abs(y).max()
                # Initial guess: amplitude, mean, stddev
                amplitude_guess = y_norm.min()
                mean_guess = x[np.argmin(y_norm)]
                stddev_guess = (x.max() - x.min()) / 8
                init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=stddev_guess)
                fitter = fitting.LevMarLSQFitter()
                fit = fitter(init, x, y_norm)
                min_x = fit.mean.value
                min_x_array[i, j, 0] = min_x
                dl[i, j, k] = min_x - lambda0[k]
                min_y = fit(min_x) * np.abs(y).max() + np.median(y)  # convert back to original scale
                # Fit quality: normalized RMSE
                residual = y - (fit(x) * np.abs(y).max() + np.median(y))
                rmse = np.sqrt(np.mean(residual**2))
                norm_rmse = rmse / (np.abs(y).max() - np.abs(y).min() + 1e-8)
                fit_quality[i, j, k] = norm_rmse
                if (verbose) or (fit_quality[i,j,k] > 0.2):
                    print(f"Minimum of Gaussian fit: x = {min_x}, fit quality = {norm_rmse:.4f}")
                    print(f"Distance: {dl[i, j, 0]}")
                    plt.plot(x, y, label='Data')
                    plt.plot(x, fit(x) * np.abs(y).max() + np.median(y), label='Gaussian fit', color='orange')
                    plt.axvline(min_x, color='green', linestyle='--', label='Fit minimum')
                    plt.axvline(lambda0[k], color='purple', linestyle='--', label='Reference line')
                    plt.legend()
                    plt.title(f'Pixel ({i},{j})')
                    plt.show()

    # Save min_x_array, fit_quality, and dl as pickle files
    with open('generated/objects/min_x_array.pickle', 'wb') as f:
        pickle.dump(min_x_array, f)
    with open('generated/objects/fit_quality.pickle', 'wb') as f:
        pickle.dump(fit_quality, f)
    with open('generated/objects/dl.pickle', 'wb') as f:
        pickle.dump(dl, f)
    with open('generated/objects/velocity.pickle', 'wb') as f:
        pickle.dump(v, f)
    print('Saved min_x_array, fit_quality, velocity, and dl to generated/* as pickle files')


else:
    v = pickle.load(open('generated/objects/velocity.pickle', "rb"))
    dl = pickle.load(open('generated/objects/dl.pickle', "rb"))
    fit_quality = pickle.load(open('generated/objects/fit_quality.pickle', "rb"))
    min_x_array = pickle.load(open('generated/objects/min_x_array.pickle', "rb"))

if True:
    for k in range(2):
        fig, _, _ = plot_data(dl[:,:,k], colourbar_label=r"$\Delta \lambda v$  [$\mathrm{\AA}$]", colourmap='bwr')#, norm=MidPointNorm(0))#, scale=[-0.08, 0.08])
        fig.savefig("generated/" + f"delta_lambda_v_{k}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/delta_lambda_v_{k}.png")

        fig, _, _ = plot_data(fit_quality[:,:,k], colourbar_label=r"RMSW", colourmap='gist_stern_r')
        fig.savefig("generated/" + f"fit_quality_{k}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/fit_quality_{k}.png")


# Calculate line-of-sight velocity
c = 2.99792458e5  # [km · s−1]
# Rebecca Centeno, equation 3
v = [c * dl[:,:,0] / lambda0[0],
     c * dl[:,:,1] / lambda0[1]]


if True:
    for k in range(2):
        fig, _, _ = plot_data(v[k], colourbar_label=r"$v_{los}$  [km/s]", colourmap='bwr')#, scale=[-0.08, 0.08])
        fig.savefig("generated/" + f"vlos_{k}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/vlos_{k}.png")



#cell 9
plate_scale_x = 0.14857 # arcseconds per pixel
plate_scale_y = 0.16 # arcseconds per pixel

# Select region of quiet sun for calibration intensity calculation
xmin = 440
xmax = I.data.shape[1]
ymin = 0
ymax = 168
xwidth = xmax - xmin
ywidth = ymax - ymin
# We want to select a region with little total polarization, since this implies a low magnetic field -> quiet sun

# Plot polarization
fig, ax, img = plot_data(I.data_n[:,:,0], colourmap='magma', title="Quiet sun region on I map")

# Create a rectangle patch and add the patch to the Axes
ax.add_patch(Rectangle((xmin*plate_scale_x, ymin*plate_scale_y), xwidth*plate_scale_x, ywidth*plate_scale_y, linewidth=2, edgecolor='r', facecolor='none'))


# Save figure to file
fig.savefig("generated/polarisation_selection_square.png", dpi=150)
print("Saved figure to file", "generated/polarisation_selection_square.png")

#cell 10
# Calulate mean velocity of quiet sun area
v_quiet_mean = [v[0][ymin:ymax,xmin:xmax].mean(),
                v[0][ymin:ymax,xmin:xmax].mean()]
print(f'Mean velocity in quiet sun: {v_quiet_mean} km/s')

# Calulate mean of fit
fit_quiet_mean = fit_quality[ymin:ymax,xmin:xmax].mean()
print(f'Mean fit quality in quiet sun: {fit_quiet_mean} RMSE')

if True:
    for k in range(2):
        fig, _, _ = plot_data(v[k] - v_quiet_mean[k], colourbar_label=r"$v_{los}$  [km/s]", colourmap='bwr')#, norm=MidPointNorm(0))#, scale=[-0.08, 0.08])
        fig.savefig("generated/" + f"vlos_norm_{k}.png", dpi=200, bbox_inches='tight')
        print("Saved figure to file", f"generated/vlos_norm_{k}.png")




#cell 11
# Plot velocity (corrected with quiet sun) in four subplots for different areas, with only one colorbar placed beside the subplots
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8.5,9))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.1)

# Define four regions as (ymin, ymax, xmin, xmax)
regions = [
    (30, 60, 65, 95),    # Penumbra (north)
    (60, 90, 0, 30),     # Quiet Sun
    (30, 60, 30, 60),    # Penumbra (south)
    (50, 80, 50, 80),    # Penumbra (west)
]

extent = [0, 89.88485, 0, 122.88]
titles = ["Penumbra (north)", "Quiet Sun", "Penumbra (south)", "Penumbra (west)"]

imgs = []
for idx, (xmin, xmax, ymin, ymax) in enumerate(regions):
    ax = fig.add_subplot(gs[idx])
    img = ax.imshow(v[0]-v_quiet_mean[0], cmap='bwr', vmin=-2, vmax=2, origin='lower', extent=extent)
    imgs.append(img)
    ax.set_title(f'{titles[idx]}')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    # Remove y axis on right figures
    if idx % 2 == 1:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('y [arcsec]')
    # Remove x axis on top row
    if idx < 2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('x [arcsec]')

# Add a single colorbar to the right of all subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(imgs[0], cax=cbar_ax, orientation='vertical', label=r'$v_{los}$ [km/s]')

# Save figure to file
fig.savefig("generated/velocity_subplots.png", dpi=200, bbox_inches='tight')
print("Saved figure to file", "generated/velocity_subplots.png")


#cell 13
delta_from_mean = [v_quiet_mean[0] / c * lambda0[0],
                     v_quiet_mean[1] / c * lambda0[1]]
print(f'Delta from mean: {delta_from_mean} Angstrom')


#cell 15
x_indices = [123, 320, 422]
y_indices = [20, 373, 685]
fig, axs = plt.subplots(3, 3, figsize=(20, 16))

for row, i in enumerate(x_indices):
    for col, j in enumerate(y_indices):
        x = I.wave_array[:60]
        y = I.data_n[j, i, :60]
        y_norm = (y - np.median(y)) / np.abs(y).max()
        amplitude_guess = y_norm.min()
        mean_guess = x[np.argmin(y_norm)]
        stddev_guess = (x.max() - x.min()) / 8
        init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=stddev_guess)
        fitter = fitting.LevMarLSQFitter()
        fit = fitter(init, x, y_norm)
        min_x = fit.mean.value
        min_y = fit(min_x) * np.abs(y).max() + np.median(y)
        residual = y - (fit(x) * np.abs(y).max() + np.median(y))
        rmse = np.sqrt(np.mean(residual**2))
        norm_rmse = rmse / (np.abs(y).max() - np.abs(y).min() + 1e-8)

        ax = axs[col, row]
        ax.plot(x, y, label='I data', linewidth=0, marker='o')
        ax.plot(x, fit(x) * np.abs(y).max() + np.median(y), label='Gaussian fit', color='orange')
        ax.axvline(min_x, color='green', linestyle='-.', label='Fit minimum')
        ax.axvline(lambda0[0], color='purple', linestyle='--', label='Reference line')
        # ax.axvline(lambda0[0] + delta_from_mean, color='black', linestyle=':', label='Delta V quiet sun')
        # ax.axvspan(lambda0[0] + delta_from_mean, lambda0[0], color='green', alpha=0.2, label='Delta V quiet sun')
        ax.set_xlim([6301.3, 6301.7])
        # ax.set_title(f'Pixel ({i},{j})\nFit min: {min_x:.4f}, RMSE: {norm_rmse:.4f}\nV: {v[j,i]-v_quiet_mean}')
        ax.set_title(f'Pixel ({i},{j})')
        ax.legend()

plt.tight_layout()
plt.show()

#cell 16
# Select pixels of interest
plate_scale_x = 0.14857 # arcseconds per pixel
plate_scale_y = 0.16 # arcseconds per pixel
x_pix = np.array([500, 180, 290, 400, 320, 300])
y_pix = np.array([118, 113, 538, 354, 428, 413])
pix_name=['A', 'B', 'C', 'D', 'E', 'F']

fig, axs = plt.subplots(3, 2, figsize=(9, 10))

for i in range(6):
    x = I.wave_array[:60]
    y = I.data_n[y_pix[i], x_pix[i], :60]
    y_norm = (y - np.median(y)) / np.abs(y).max()
    amplitude_guess = y_norm.min()
    mean_guess = x[np.argmin(y_norm)]
    stddev_guess = (x.max() - x.min()) / 8
    init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=stddev_guess)
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(init, x, y_norm)
    min_x = fit.mean.value
    min_y = fit(min_x) * np.abs(y).max() + np.median(y)
    residual = y - (fit(x) * np.abs(y).max() + np.median(y))
    rmse = np.sqrt(np.mean(residual**2))
    norm_rmse = rmse / (np.abs(y).max() - np.abs(y).min() + 1e-8)
    row, col = divmod(i, 2)
    ax = axs[row, col]
    ax.plot(x, y, label='I data', linewidth=0, marker='.')
    ax.plot(x, fit(x) * np.abs(y).max() + np.median(y), label='Gaussian fit', color='orange')
    ax.axvline(min_x, color='green', linestyle='-.', label='Fit minimum')
    ax.axvline(lambda0[0], color='purple', linestyle='--', label='Reference line')
    # ax.axvline(lambda0[0] + delta_from_mean, color='black', linestyle=':', label='Delta V quiet sun')
    # ax.axvspan(lambda0[0] + delta_from_mean, lambda0[0], color='green'        , alpha=0.2, label='Delta V quiet sun')
    ax.set_xlim([6301.2, 6301.8])
    ax.set_title(f'Pixel {pix_name[i]}\nRMSW: {norm_rmse:.4f}')
    ax.legend()

#cell 19
# Fit a Gaussian to the data and find its minimum
verbose = 0

# shape = np.shape(I.data_n[:,:,:2])
# fit_quality = np.zeros(shape)
# min_x_array = np.zeros(shape)
# dl = np.zeros(shape)

for i in range(0, shape[0]):
    if np.mod(i,10) == 0:
        print(f'Row {i} of {np.shape(I.data_n)[0]}')
    for j in range(0, shape[1]):
        if (fit_quality[i,j,0]>0.075):
            fit_quality[i,j,0] = None
            x = I.wave_array[:60]
            y = copy.copy(V.data_n[i,j,:60])

            # First line
            peaks_p = int(np.argmax(y[:60]))
            peaks_n = int(np.argmin(y[:60]))

            min_x = 0.5*(V.wave_array[peaks_p] - V.wave_array[peaks_n]) + V.wave_array[peaks_n]

            min_x_array[i, j, 0] = min_x
            dl[i, j, 0] = min_x - lambda0[0]
            v1 = c * dl[i,j,0] / lambda0[0]

            if (verbose):
                print(f"Minimum of Gaussian fit: x = {min_x}")
                print(f"Distance: {dl[i, j, 0]}, {min_x - lambda0[0]}")
                plt.plot(x, y, label='Data')
                plt.axvline(V.wave_array[peaks_p], color='black', linestyle=':')
                plt.axvline(V.wave_array[peaks_n], color='black', linestyle=':')
                plt.axvline(min_x, color='green', linestyle='--', label='Halfway between peaks')
                plt.axvline(lambda0[0], color='purple', linestyle='--', label='Reference line')
                plt.legend()
                plt.title(f'Pixel ({i},{j})\nFit min: {min_x:.4f}, V: {v1 -v_quiet_mean}')
                plt.show()



#cell 20
# # For umbra pixels find delta lambda with the peaks of Stokes V
# verbose = 0

# # shape = np.shape(I.data_n[:,:,:2])
# # fit_quality = np.zeros(shape)
# # min_x_array = np.zeros(shape)
# # dl = np.zeros(shape)

# for i in range(0, shape[0]):
#     if np.mod(i,10) == 0:
#         print(f'Row {i} of {np.shape(I.data_n)[0]}')
#     for j in range(0, shape[1]):
#         if np.isnan(fit_quality[i,j,0]):
#             x = I.wave_array[:60]
#             y = copy.copy(V.data_n[i,j,:60])

#             # First line
#             peaks_p = int(np.argmax(y[:60]))
#             peaks_n = int(np.argmin(y[:60]))

#             min_x = 0.5*(V.wave_array[peaks_p] - V.wave_array[peaks_n]) + V.wave_array[peaks_n]

#             min_x_array[i, j, 0] = min_x
#             dl[i, j, 0] = min_x - lambda0[0]
#             v1 = c * dl[i,j,0] / lambda0[0]
#             print(f"V: {v1}")

#             if (verbose or np.abs(v1)>1.5):
#                 print(f"Minimum of Gaussian fit: x = {min_x}")
#                 print(f"Distance: {dl[i, j, 0]}, {min_x - lambda0[0]}")
#                 plt.plot(x, y, label='Data')
#                 plt.axvline(V.wave_array[peaks_p], color='black', linestyle=':')
#                 plt.axvline(V.wave_array[peaks_n], color='black', linestyle=':')
#                 plt.axvline(min_x, color='green', linestyle='--', label='Halfway between peaks')
#                 plt.axvline(lambda0[0], color='purple', linestyle='--', label='Reference line')
#                 plt.legend()
#                 plt.title(f'Pixel ({i},{j})\nFit min: {min_x:.4f}, V: {v1 -v_quiet_mean}')
#                 plt.show()


#cell 21
# Rebecca Centeno, equation 3
v = c * dl[:,:,0] / lambda0[0]

fig,ax,img = plot_data(v-v_quiet_mean, colourbar_label=f"Velocity [km/s]", colourmap='bwr', scale=[-2, 2])
ax.set_xlim([25, 60])
ax.set_ylim([55, 77])

