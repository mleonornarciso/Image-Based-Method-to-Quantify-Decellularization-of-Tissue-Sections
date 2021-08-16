import img_processing

# -----Parameters that can be tuned:
comp_1=2000
comp_2=10000
#these parameters correspond to the pixel size of the components that will be eliminated from the mask
#depending on image resolution and biological properties of the tissue, these can be optimized
#comp_2 corresponds to the size of components corresponding to artifacts outside the mask

bg_inten = []

# -----create an image pair formed of the phase contrast image and the corresponding uv image with the stained DNA 
pc_img = img_pair[0]
uv_img = img_pair[1]

#mask pre processing (rgb to gray scale, contrast stretching and normalization)
pc_gs = img_processing.rgb2gray(pc_img)
pc_eq = img_processing.eq_norm(pc_gs)
#apply entropy and threshold to create the mask
mask_1,ent_img= img_processing.entropy_filter(pc_eq)
#dilations and erosions to close connected components, performs 3 dilations and 3+1 erosions
mask_holes=img_processing.dil_er(mask_1,3,3)
#remove connected components that are smaller than comp_1
mask_labeled,ncomponents1=img_processing.mask_labeling(img_processing.invert_mask(mask_holes))
mask_hole_free = img_processing.invert_mask(img_processing.remove_small_components(mask_labeled, ncomponents1,comp_1))
#remove artifacts and noise - components outside the mask smaller than comp_2
#achieve the final mask
mask_labeled_2, ncomponents_2 = img_processing.mask_labeling(mask_hole_free) 
mask=img_processing.remove_small_components(mask_labeled_2,ncomponents_2,comp_2)        
mask_size = mask.sum()

#apply mask to uv image to obtain only pixels of interest corresponding to tissue
masked_uv = img_processing.apply_mask2img(uv_img,mask)
#apply inverted mask to uv image to obtain only background pixels
masked_bg = img_processing.apply_mask2img(uv_img,img_processing.invert_mask(mask))

#compute median intensity of the pixels of interest
uv_intensity_masked, intensity_total, nb_pixels_uv = img_processing.average_intensity(masked_uv,1,mask.sum())
#compute the median intensity of the background pixels
bg_intensity, intensity_total_bg, nb_pixels_bg = img_processing.average_intensity(masked_bg,2,img_processing.invert_mask(mask).sum())
#reach the final intensity by subtracting the background intensity
final_avg_intensity = uv_intensity_masked-bg_intensity

