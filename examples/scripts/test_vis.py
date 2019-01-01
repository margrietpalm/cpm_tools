from cpm_tools import *

w = 100
h = 100
ncells = 25
r = 5

# seed pixels
grid = seed_cells(w,h,ncells,pad=2*r,dist=2*r)

# add pixels to each seed to make circles
grid = grow_cells_round(grid,r)
tau = np.zeros_like(grid)
tau[grid>0] = 1
colormap = {0:(255,255,255),1:(255,0,0)}
# save to tiff

draw_cpm_grid(grid,tau,colormap,'test.png',scale=5)
add_text('test.png','top center',(.5,1))
add_text('test.png','bottom center',(.5,0))
add_text('test.png','top left',(0,1))
add_text('test.png','top right',(1,1))
