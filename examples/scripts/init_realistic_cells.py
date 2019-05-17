from cpm_tools import *

w = 100
h = 100
ncells = 25
volume = 80

# seed pixels
grid = seed_cells(w,h,ncells,pad=5)

# add pixels to each seed to make circles
grid = grow_cells_DLA(grid,volume)

# save to tiff
write_to_tiff(grid,'../images/cells_DLA.tiff')
