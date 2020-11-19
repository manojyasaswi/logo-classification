# %% 
import Augmentor
import os

def augment_fn(dir_str):
    p = Augmentor.Pipeline(dir_str)
    #p.random_brightness(probability = 0.5, min_factor = 0.6, max_factor = 1.0)
    #p.random_color(probability =0.5, min_factor = 0.2, max_factor = 0.8)
    #p.random_contrast(probability =0.5, min_factor = 0.5, max_factor = 0.8)
    p.skew(probability = 0.4, magnitude= 0.2)
    p.shear(probability = 0.3, 
    max_shear_left=5, max_shear_right=5)
    p.rotate(probability = 0.5,
    max_left_rotation= 15, max_right_rotation=15)
    p.rotate90(probability = 0.5)
    p.rotate180(probability = 0.5)
    p.rotate270(probability = 0.5)
    #p.random_distortion(probability = 0.2,
    #grid_height = 10, grid_width =10, magnitude= 2)
    p.zoom(probability=0.2, min_factor=1.1, max_factor=1.15)
    p.sample(1000)


# %%
src = './data/Logos'

for (root,dirs,files) in os.walk(src):
    for i in range(len(dirs)):
        dir_str = src+'/'+dirs[i]
        #print(dir_str)
        #l = os.listdir(dir_str)
        #if len(l) <1:
        #    print(dir_str)
        #print(len(l))
        augment_fn(dir_str)
        print("\n Processing Logo: ",str(i),"\n")
        


#
# %%

# %%
