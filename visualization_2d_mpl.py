import future        # pip install future
import matplotlib.pyplot as plt
import numpy as np
import copy,os
import matplotlib.colors as mpl_colors
import matplotlib.animation as animation


def animate_cpm_sim_mpl(tau_list,colors,fn=None,show=False,dpi=100,scale=1,fps=5):
    """ Animate cpm simulation using matplotlib

    :param tau_list: list of grids with tau
    :param colors: list of matplotlib colornames, in the order of tau (see https://matplotlib.org/examples/color/named_colors.html)
    :param fn: filename used for saving the video
    :param show: show animation
    :param dpi: dpi
    :param scale: scaling factor
    :param fps: animation frame rate
    """

    # setup figure
    fig = plt.figure(figsize=(scale * tau_list[0].shape[0] / 100., scale * tau_list[0].shape[1] / 100.),dpi=dpi)
    ax = plt.gca()

    # setup colormap
    cmap = mpl_colors.ListedColormap(colors)
    bounds = np.arange(0, 7)
    norm = mpl_colors.BoundaryNorm(bounds, cmap.N)

    # create images
    ims = []
    for tau in tau_list:
        im = plt.imshow(tau, interpolation='nearest', origin='lower',
                        cmap=cmap, norm=norm, animated=True)
        ims.append([im])
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # create animation
    anim = animation.ArtistAnimation(fig, ims, interval=1000./fps, blit=True,repeat_delay=1000)
    if show:
        plt.show()
    if fn is not None:
        anim.save(fn, fps=fps, extra_args=['-vcodec', 'libx264'])
    return anim


def draw_cpm_grid_mpl(tau,colors,ax=None,fn=None,dpi=100,scale=1):
    """ Draw cpm grid using matplotlib.

    Drawing the cpm grid using matplotlib is very fast, but does not support drawing
    cell borders.

    :param tau: array with cell types
    :param colors: list of matplotlib colornames, in the order of tau (see https://matplotlib.org/examples/color/named_colors.html)
    :param ax: matplotlib ax to use
    :param fn: filename used for saving
    :param dpi: output image dpi
    :param scale: scaling factor
    """

    # create axis object
    if ax is None:
        fig = plt.figure(figsize=(scale*tau.shape[0]/float(dpi),scale*tau.shape[1]/float(dpi)))
        ax = plt.gca()

    # setup color map
    cmap = mpl_colors.ListedColormap(colors)

