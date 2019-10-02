import copy
import numpy as np
import imageio
import numba

def grow_cells_DLA(grid, volume):
    """
    Grow cells using diffusion limited aggregation (DLA). Growth continues until
    all cells have the desired volume and cells cannot grow beyond that volume.

    Args:
        w: CPM grid
        volume: cell volume

    Returns: CPM grid

    """
    n = np.max(grid)
    w = grid.shape[0]
    h = grid.shape[1]
    nx = [-1, -1, 0, 1, 1, 1, 0, -1]
    ny = [0, 1, 1, 1, 0, -1, -1, -1]
    pix = [[x, y] for x in range(w) for y in range(h)]
    # while np.any(np.bincount(grid.flatten())[1:] < volume):
    while np.sum(grid > 0) < volume * n:
        grid = _DLA_step(copy.deepcopy(grid),volume,w,h)
    return grid


@jit(nopython=True)
def _DLA_step(grid,volume,w,h):
    nx = [-1, -1, 0, 1, 1, 1, 0, -1]
    ny = [0, 1, 1, 1, 0, -1, -1, -1]
    pix = [[x, y] for x in range(w) for y in range(h)]
    r = np.random.randint(0, 7, w * h)
    for i, (x, y) in enumerate(pix):
        if grid[x, y] > 0:
            continue
        nb_x = x + nx[r[i]]
        nb_y = y + ny[r[i]]
        if (nb_x < 0) or (nb_y < 0) or (nb_x >= w) or (nb_y >= h):
            continue
        if grid[nb_x, nb_y] > 0 and np.sum(grid == grid[nb_x, nb_y]) < volume:
            grid[x, y] = grid[nb_x, nb_y]
    return grid


def grow_cells_round(grid, r):
    """
    Grow seeded cells into circles

    Args:

        grid: CPM grid with seeded cells
        r: cell radius

    Returns: Numpy array representing sigma

    """
    pix = np.column_stack(np.where(grid > 0))
    w = grid.shape[0]
    h = grid.shape[1]
    for idx, (x, y) in enumerate(pix, 1):
        grid = _grow_to_circle(grid,idx,x,y,r,w,h)
    return grid


@jit(nopython=True)
def _grow_to_circle(grid,idx,x,y,r,w,h):
    for i in range(-r, r):
        for j in range(-r, r):
            if (x + i > 0) and (y + j > 0) and (x + i < w) and (y + j < h):
                if (i ** 2 + j ** 2) < r ** 2:
                    grid[x + i, y + j] = idx
    return grid


def seed_cells(w, h, n, pad=10, dist=0, maxit=1000):
    """
    Randomly place single pixels on the CPM grid. When the minimum distance is zero, cells are placed randomly
    without considering the position of other cells. When the minimum distance is larger than zero,
    the algorithm attempts to position the desired number of cells considering the minimum distance. To prevent
    an infinite run, the algorithm is stopped when the maximum number of iterations is reached.

    Args:
        w: grid width
        h: grid height
        n: number of cells
        pad: padding between cells and border
        dist: minimum distance between cells
        maxit: maximum number of iterations per cells

    Returns: CPM grid

    """
    if dist > 0:
        return _seed_cells_complicated(w, h, n, pad, dist, maxit)
    else:
        return _seed_cells_naive(w, h, n, pad)


def _seed_cells_naive(w, h, n, pad=10):
    grid = np.zeros((w, h))
    if w * h < n:
        pix = [(i, j) for i in range(w) for j in range(h)]
        for idx, (i, j) in enumerate(pix, 1):
            grid[i, j] = idx
    else:
        while np.sum(grid > 0) < n:
            x0 = np.random.randint(pad, w - pad)
            y0 = np.random.randint(pad, h - pad)
            if grid[x0, y0] == 0:
                grid[x0, y0] = np.sum(grid > 0) + 1
    return grid


def _seed_cells_complicated(w, h, n, pad=10, dist=2, maxit=1000):
    pix = []
    it = 0
    while len(pix) < n:
        x = np.random.randint(pad, w - pad)
        y = np.random.randint(pad, h - pad)
        add = True
        for p in pix:
            if ((p[0] - x) ** 2 + (p[1] - y) ** 2) < dist ** 2:
                add = False
                break
        if add:
            pix.append((x, y))
        it += 1
        if it > maxit * n:
            break
    grid = np.zeros((w, h))
    for i, (x, y) in enumerate(pix, 1):
        grid[x, y] = i
    return grid


def write_to_tiff(grid, fn):
    """
    Save grid to tiff.

    Args:
        grid: CPM grid
        fn: filename

    """
    if grid.max() < 256:
        imageio.imwrite(fn, grid.astype(np.uint8), format='tiff')
    else:
        imageio.imwrite(fn, grid.astype(np.uint16), format='tiff')

