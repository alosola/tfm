# Define B&W plot function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_data(data, title='', scale=None, norm=None, colourbar_label=None, colourmap='magma', size='normal'):
    # Plot one frame of data
    if size=='small':
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,4))
    elif size=='large':
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,8))
    else:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    ax.set_xlabel('x [arcsec]')
    ax.set_ylabel('y [arcsec]')
    ax.set_title(title)

    # plate_scale_x = 0.14857 # arcseconds per pixel
    # plate_scale_x = 0.16 # arcseconds per pixel
    # extent = [0, stokes_list['I'].size_x*plate_scale_x, 0, stokes_list['I'].size_y*plate_scale_y]
    extent=[0, 89.88485, 0, 122.88]

    if (scale==None):
        # If no scale provided, use default
        img = ax.imshow(data, cmap=colourmap, norm=norm, origin='lower', extent=extent)
    else:
        img = ax.imshow(data, cmap=colourmap, vmin = scale[0], vmax = scale[1], norm=norm, origin='lower', extent=extent)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label=colourbar_label)

    return fig, ax, img


# Colourmap cheat sheet:
# - polarisation: gist_earth
# - velocity: bwr
# - intensity, temperature: magma
# - Q, U, V:  'PuOr_r'(divnorm)