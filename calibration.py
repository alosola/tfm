## Data calibration
# - calculate quiet sun mean profile
# - load calibration spectrum
# - find peaks in calibration and data spectrum
# - polyfit to find calibrated wavelength array
# - save updated Stokes object


from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import copy
import pickle

# Project modules
from lib.Stokes import Stokes


# Check that the Stokes object file exists
stokes_filename = "generated/objects/stokes.pickle"
my_file = Path(stokes_filename)
if not my_file.is_file():
    print("Stokes object does not exists (", stokes_filename,"), please run file init_data.py first")

# Load the object from the file
print("Loading Stokes data from file", stokes_filename)
stokes_list = pickle.load(open(stokes_filename, "rb"))

# Select region of quiet sun for calibration intensity calculation
xmin = 0
xmax = 180
ymin = 450
ymax = 605
xwidth = xmax - xmin
ywidth = ymax - ymin
# We want to select a region with little total polarization, since this implies a low magnetic field -> quiet sun

# Calulate mean value of quiet sun area for each wavelength
for param in stokes_list:
    stokes_list[param].mean_quiet_region(xmin, xmax, ymin, ymax)

# Open calibration data
calibfile = 'data/fts_calibration.npz'
print("Reading calibration data file", calibfile)
calibdata = np.load(calibfile)
print("Opened calibration data file")

# Print data parameters
print("List of keys in file:", list(calibdata.files))
key = list(calibdata.files)
print("With shape:")
print("    Wavelength daya (x):           ", calibdata[key[0]].shape)
print("    Intensity (y):                 ", calibdata[key[0]].shape)
print("    Continuum (c):                 ", calibdata[key[0]].shape)

# Get spectral lines from data spectrum
spectrum = copy.copy(stokes_list['I'].mean_quiet)
peaks, _ = find_peaks(-spectrum) 
print("Peaks in I data: ", peaks)

# Example from here: https://eikonaloptics.com/blogs/tutorials/spectrometer-wavelength-calibration-practical-implementation?srsltid=AfmBOoqBsKn0cOmwJ4wTow4yGllnfrRJAqNRn0FOSJ3sFu7leDetbL1D
# Find centroid of spectral lines
npix = 4
centroid_pix = np.array([])
for p in peaks:
  pix = np.arange(p-npix, p+npix+1)
  centroid_pix = np.append(centroid_pix,
                 np.sum(spectrum[p-npix: p+npix+1] * pix) / np.sum(spectrum[p-npix: p+npix+1]))
  
print("Centroix pixels on spectrum: ", centroid_pix)

# Get spectral lines from calibration spectrum
calib_spectrum = calibdata['y']
calib_peaks, _ = find_peaks(-calib_spectrum) # find absortion lines
print("Peaks in calibration data: ", calib_peaks)
# Keep peaks which match the Fe I lines (manually)
calib_peaks_clean = [210, 347]

# Get wavelength values of Fe I peaks
calib_wavelengths = [calibdata['x'][210], calibdata['x'][347]]
print("Wavelengths of Fe I peaks: ", calib_wavelengths)

# Calculate polynomial which fits the spectra to the calibration wavelengths
poly_degree = 1
coeffs_wave_cal = np.polyfit(centroid_pix, calib_wavelengths, deg=poly_degree, w=[20,2])
print("Polyfit coefficients: ", coeffs_wave_cal)

# Adjust spectrum with the calculated polynomial
pix_val = np.arange(len(spectrum))
calibrated_axis = np.polyval(coeffs_wave_cal, pix_val)

# Add wavelength axis to objects
for param in stokes_list:
    stokes_list[param].add_wavelength(calibrated_axis)

    
# Save updated Stokes objects
stokes_filename = "generated/objects/stokes.pickle"
print("Saving datacube to pickle file:", stokes_filename)
with open(stokes_filename, 'wb') as handle:
    pickle.dump(stokes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)