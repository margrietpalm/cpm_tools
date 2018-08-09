import numpy as np
import copy,os


try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise('cannot draw without PIL')

__FONTPATH__ = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'



def get_2d_projection(sigma,tau,projection):
    """ Draw 2D projection of a 3D cpm simulation

    :param sigma: 3D array with cell ids
    :param tau: 3D array with cell types
    :param projection: string with projection plane (x, y, or z) and direction ('+' = top and '-' = bottom)
    """
    top = -1
    if '-' in projection:
        top = 0
    if 'x' in projection:
        sigma_2d = np.array([[sigma[:, y, z][sigma[:, y, z] > 0][top] if np.any(sigma[:, y, z] > 0) else 0
                              for z in range(sigma.shape[2])] for z in range(sigma.shape[1])])
    elif 'y' in projection:
        sigma_2d = np.array([[sigma[x, :, z][sigma[x, :, z] > 0][top] if np.any(sigma[x, :, z] > 0) else 0
                              for z in range(sigma.shape[2])] for x in range(sigma.shape[0])])
    elif 'z' in projection:
        sigma_2d = np.array([[sigma[x, y, :][sigma[x, y, :] > 0][top] if np.any(sigma[x, y, :] > 0) else 0
                          for y in range(sigma.shape[1])] for x in range(sigma.shape[0])])
    else:
        print('Unrecognized projection {}'.format(projection))
        return
    # map tau
    tau_2d = np.zeros_like(sigma_2d)
    for s in np.unique(sigma_2d):
        px, py, pz = np.array(np.where(sigma == s))[:, 0]
        tau_2d[np.where(sigma_2d == s)] = tau[px, py, pz]
    return sigma_2d,tau_2d


def draw_2d_projection(sigma,tau,colormap,fn,projection='+z',scale=1,border_color=None,draw_border=True):
    """ Draw 2D projection of a 3D cpm simulation

    :param sigma: 3D array with cell ids
    :param tau: 3D array with cell types
    :param colormap: dictionary with tau as keys and colors (rgb tuples) as values
    :param fn: filename used for saving
    :param projection: string with projection plane (x, y, or z) and direction ('+' = top and '-' = bottom)
    :param scale: image scaling
    :param border_color: color of the cell borders
    :param draw_border: draw cell borders
    """
    sigma_2d,tau_2d = get_2d_projection(sigma,tau,projection)
    draw_cpm_grid(sigma_2d,tau_2d,colormap,fn,scale,border_color,draw_border)


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
        from mahotas import labeled
    except ImportError:
        print('Drawing borders is only available when mahotas is installed!')
        draw_border = False
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
    imnew = colormap[0]*np.ones((scale*nx,scale*ny,len(colormap[0])))
    if np.sum(sigma) > 0:
        for tp in np.unique(types):
            if tp == 0:
                continue
            imnew[types==tp] = colormap[tp]
        imnew[bim] = border_color

    # save final image
    final_im = Image.fromarray(np.uint8(imnew))
    final_im.save(fn)


def add_legend(imname, colormap, wbox=10, hbox=10, fontcolor=(0, 0, 0),
               fontpath=__FONTPATH__,fontsize=14, outname=None):
    font = ImageFont.truetype(fontpath, fontsize)
    # get label height
    temp = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(temp)
    (tw,th) = draw.textsize(str(list(colormap.keys())[0]), font=font)
    # add labels
    im = Image.open(imname)
    draw = ImageDraw.Draw(im)
    (w,h) = im.size
    x0 = 10
    y0 = 10
    dh = hbox+5
    for i,(name,color) in enumerate(colormap.items()):
        draw.rectangle([(x0,y0+i*dh),(x0+wbox,y0+i*dh+hbox)],fill=color,outline=(0,0,0))
        draw.text((x0+wbox+5,y0+i*dh+.5*hbox-.5*th),str(name),fill=fontcolor,font=font)
    if outname is None:
        outname = imname
    im.save(outname)


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


