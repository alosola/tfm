# Define B&W plot function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_angle_gradient(data_radians, title='', colourmap='twilight', colourbar_label='Angle [degrees]', scale=''):
    # Plot angle with cyclic colormap
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.set_xlabel('X axis (array index)')
    ax.set_ylabel('Y axis (array index)')
    ax.set_title(title)

    # plate_scale_x = 0.14857 # arcseconds per pixel
    # plate_scale_x = 0.16 # arcseconds per pixel
    # extent = [0, stokes_list['I'].size_x*plate_scale_x, 0, stokes_list['I'].size_y*plate_scale_y]
    extent=[0, 89.88485, 0, 122.88]

    if scale == '':
        img = ax.imshow(np.rad2deg(data_radians), cmap=colourmap, origin='lower', extent=extent)
    else:
        img = ax.imshow(np.rad2deg(data_radians), cmap=colourmap, origin='lower', vmin=scale[0], vmax=scale[1], extent=extent)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label=colourbar_label)
    plt.show()

    return fig, ax