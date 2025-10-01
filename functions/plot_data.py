# Define B&W plot function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_data(data, title='', scale=None, norm=None, colourbar_label=None, colourmap='magma'):
    # Plot one frame of data
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.set_xlabel('X axis (array index)')
    ax.set_ylabel('Y axis (array index)')
    ax.set_title(title)

    # plate_scale_x = 0.14857 # arcseconds per pixel
    # plate_scale_x = 0.16 # arcseconds per pixel
    # extent = [0, I.size_x*plate_scale, 0, I.size_y*plate_scale]

    if (scale==None):
        # If no scale provided, use default
        img = ax.imshow(data, cmap=colourmap, norm=norm, origin='lower', extent=[0, 96.8, 0, 122.88])
    else:
        img = ax.imshow(data, cmap=colourmap, vmin = scale[0], vmax = scale[1], norm=norm, origin='lower', extent=[0, 96.8, 0, 122.88])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label=colourbar_label)

    return fig, ax