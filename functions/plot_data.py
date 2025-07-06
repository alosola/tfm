# Define B&W plot function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_data(data, title):
    # Plot one frame of data
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.set_xlabel('X axis (array index)')
    ax.set_ylabel('Y axis (array index)')
    ax.set_title(title)
    img = ax.imshow(data, cmap='gray', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, label='Number of counts')
    plt.show()

    return fig, ax