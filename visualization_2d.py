import future        # pip install future
import matplotlib.pyplot as plt
import numpy as np
import copy,os
import matplotlib.colors as mpl_colors
import matplotlib.animation as animation

__FONTPATH__ = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'


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
    bounds=np.array(list(np.unique(tau))+[np.max(tau)+1])
    norm = mpl_colors.BoundaryNorm(bounds, cmap.N)

    # draw image
    img = plt.imshow(tau, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')

    # save image
    if fn is not None:
        plt.savefig(fn,dpi=dpi)
        plt.close()

def draw_cpm_grid(sigma,tau,colormap,fn,scale=1,border_color=None,draw_border=True):
    """ Draw cpm grid

    Draw cpm grid with any level of magnification and cell borders.

    :param sigma: array with cell ids
    :param tau: array with cell types
    :param colormap: dictionary with tau as keys and colors (rgb tuples) as values
    :param fn: filename used for saving
    :param scale: image scaling
    :param border_color: color of the cell borders
    :param draw_border: draw cell borders
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        from mahotas import labeled
    except ImportError:
        try:
            from mahotas import labeled
        except ImportError:
            print('drawing borders is only available when mahotas is installed.')
            draw_border = False
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print('cannot draw without PIL')
            return

    if border_color is None:
        border_color = (0,0,0)
    (nx,ny) = sigma.shape
    sigma = sigma.astype(np.float)
    types = tau.astype(np.float)

    # return empty image if sigma is empty
    if np.sum(sigma) == 0:
        im = Image.fromarray(255*np.uint8(np.ones_like(sigma)))
        im = im.resize((int(scale*ny),int(scale*nx)))
        return np.asarray(im),sigma,np.zeros_like(sigma)

    # resize sigma and tau
    im = Image.fromarray(sigma)
    im = im.resize((int(scale*ny),int(scale*nx)))
    sigma = np.asarray(im)
    tim = Image.fromarray(np.uint8(types))
    tim = tim.resize((int(scale*ny),int(scale*nx)))
    types = np.asarray(tim)

    # used mahotas labeled function to retrieve the borders
    if draw_border:
        sigMin = 0
        bim = None
        # Labeled gives each component an integer value between 0 and 255.
        # Hence, the algorithm fails when there are more than 255 cells.
        # Therefore, we retrieve the borders per 255 cells.
        while sigMin < np.max(sigma):
            imtemp = copy.deepcopy(np.asarray(im))
            imtemp[np.where(imtemp < sigMin)] = 0
            imtemp[np.where(imtemp > sigMin+255)] = 0
            if bim is None:
                bim = labeled.borders(imtemp)
            else:
                bim += labeled.borders(imtemp)
            sigMin += 255
    else:
        bim = []

    # combine the components created above into one image
    imnew = colormap[0]*np.ones((scale*nx,scale*ny,3))
    if np.sum(sigma) > 0:
        for tp in np.unique(types):
            if tp == 0:
                continue
            imnew[types==tp] = colormap[tp]
            imnew[bim] = border_color

    # save final image
    final_im = Image.fromarray(np.uint8(imnew))
    final_im.save(fn)


def add_color_bar(imname, colors, labels, w, h, fontcolor=(0, 0, 0), bgcolor=(255, 255, 255), fontpath=__FONTPATH__,
                fontsize=24, outname=None, horizontal=False, title=None):
    """ Add colorbar to an image

    :param imname: image filename
    :param colors: list of colors
    :param labels: list of labels
    :param w: width of the colorbar
    :param h: height of the colorbar
    :param fontcolor: font color (r,g,b)
    :param bgcolor: background color (r,g,b)
    :param fontpath: path to font
    :param fontsize: font size
    :param outname: name of the new image
    """
    im = Image.open(imname)
    if horizontal:
        im = _add_color_bar_horizontal(im, colors, w, h, labels, fontcolor, bgcolor, fontpath, fontsize)
    else:
        im = _add_color_bar_vertical(im, colors, w, h, labels, fontcolor, bgcolor, fontpath, fontsize)
    if outname is None:
        im.save(imname)
    else:
        im.save(outname)

def _add_color_bar_vertical(im,colors,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),
                         fontpath=__FONTPATH__,fontsize=24):
    if labels is not None:
        font = ImageFont.truetype(fontpath, fontsize)
        lablen = [len(label) for label in labels]
        lbig = labels[lablen.index(max(lablen))]
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        tsize = draw.textsize(str(lbig), font=font)
        h = h-tsize[1]
        w = w-1.1*tsize[0]-10
    nx = int(math.ceil(w))
    ny = int(math.ceil(h))
    newim = Image.new('RGB',(nx,ny),bgcolor)
    dh = h/float(len(colors))
    draw = ImageDraw.Draw(newim)
    y0 = ny-(ny-h)/2.
    for idx,c in enumerate(colors):
        color = tuple(int(255*c[i]) for i in [0,1,2])
        draw.rectangle([(0,y0-idx*dh),(w,y0-(idx+1)*dh)],fill=color,outline=color)
    if labels is not None:
        x = w+0.1*tsize[0]
        for i,label in enumerate(labels):
            y = y0-.5*tsize[1]-h*i/float(len(labels)-1)
            draw.text((x,y),str(label),fill=fontcolor,font=font)
    im.paste(newim,((im.size[0]-nx)-nx/2,(im.size[1]-ny)/2))
    return im

def _add_color_bar_horizontal(im,colors,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),
                         fontpath=__FONTPATH__,fontsize=24):
    if labels is not None:
        font = ImageFont.truetype(fontpath, fontsize)
        lablen = [len(label) for label in labels]
        lbig = labels[lablen.index(max(lablen))]
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        tsize = draw.textsize(str(lbig), font=font)
        h = h-tsize[1]
        w = w-1.1*tsize[0]-10
    nx = int(math.ceil(w))
    ny = int(math.ceil(h))
    newim = Image.new('RGB',(nx,ny),bgcolor)
    dh = h/float(len(colors))
    draw = ImageDraw.Draw(newim)
    y0 = ny-(ny-h)/2.
    for idx,c in enumerate(colors):
        color = tuple(int(255*c[i]) for i in [0,1,2])
        draw.rectangle([(0,y0-idx*dh),(w,y0-(idx+1)*dh)],fill=color,outline=color)
    if labels is not None:
        x = w+0.1*tsize[0]
        for i,label in enumerate(labels):
            y = y0-.5*tsize[1]-h*i/float(len(labels)-1)
            draw.text((x,y),str(label),fill=fontcolor,font=font)
    im.paste(newim,((im.size[0]-nx)-nx/2,(im.size[1]-ny)/2))
    return im