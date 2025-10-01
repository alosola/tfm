# Define B&W plot function

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_angle_gradient(data_radians, title='', colourmap='twilight'):
    # Plot angle with cyclic colormap
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.set_xlabel('X axis (array index)')
    ax.set_ylabel('Y axis (array index)')
    ax.set_title(title)

    img = ax.imshow(np.rad2deg(data_radians), cmap=colourmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label='Angle [degrees]')
    plt.show()

    return fig, ax