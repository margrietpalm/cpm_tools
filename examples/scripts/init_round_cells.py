from cpm_tools import *

w = 100
h = 100
ncells = 25
r = 5

# seed pixels
grid = seed_cells(w,h,ncells,pad=2*r,dist=2*r)

# add pixels to each seed to make circles
grid = grow_cells_round(grid,r)

# save to tiff
write_to_tiff(grid,'../images/cells_round.tiff')
