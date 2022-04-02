import matplotlib.pyplot as plt

# Figure Dimensions

fig_width_in = 5.5
fig_height_in = fig_width_in
top_margin_in = 1
bottom_margin_in = 1
left_margin_in = 1
right_margin_in = 1
gap_in = 0.5

# Map
mheight_in = 3
mwidth_in = 3
mleft_in = 1
mbottom_in = fig_width_in - mheight_in
map_bounds = [mleft_in/fig_width_in, mbottom_in/fig_height_in, mwidth_in/fig_width_in, mheight_in/fig_height_in]

# Cross Section Profile
mleft_in = mleft_in
mwidth_in = mwidth_in
mheight_in = 1
mbottom_in = mbottom_in - gap_in - mheight_in
xs_bounds = [mleft_in/fig_width_in, mbottom_in/fig_height_in, mwidth_in/fig_width_in, mheight_in/fig_height_in]

# Colorbar Profile
mleft_in = 1+3+gap_in
mbottom_in = mbottom_in
mheight_in = 3+gap_in+mheight_in
mwidth_in = 0.25
cb_bounds = [mleft_in/fig_width_in, mbottom_in/fig_height_in, mwidth_in/fig_width_in, mheight_in/fig_height_in]

