import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import threshold_yen


def rgb2gray(rgb_img):
	"""
	Convert image from rgb to greyscale. 
	"""
	gray = (0.3*rgb_img[:,:,0])+(0.59*rgb_img[:,:,1])+(0.11*rgb_img[:,:,2])

	return gray


def eq_norm(img):
	
	"""
	Normalizes and equalized a grayscale image
	shifts image to 0 to 1, increases contrast by setting the max to 1 and min to -1
	eliminates outlier pixels, rescaling between 2 and 98% percentiles: increasing contrast 
	"""
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
	img_eq_norm=img_rescale/img_rescale.max()

	return img_eq_norm


def entropy_filter (img):
	"""
	Applies an entropy filter to an image and thresolds it (yen thresholding)
	Input: normalized and equalized image
	Returns a masked image based on the entropy of the image.
	"""
	ent_img=entropy(img, disk(20))
	thresh=threshold_yen(ent_img)
	thresh_img = ent_img <= thresh
	mask = invert_mask(thresh_img.astype(int))

	return mask,ent_img


def dil_er(mask,kernel,it):
	
	"""
	it Dilations followed by an it+1 erosions. Transforms the input into uint8
	Useful to disconnect holes that have been brought together so
	the labeling and hole removal works better. 
	"""
	
	mask_i= mask.astype(np.uint8)
	kernel_dil = disk(kernel)
	kernel_er = disk(kernel)
	dilation = cv2.dilate(mask_i,kernel_dil,iterations = it)
	erosion = cv2.erode(dilation,kernel_er,iterations = it+1)

	return erosion


def mask_labeling(mask):
	
	"""
	Uses the label function from scipy to categorize the different components according to pixel size.
	"""		
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    return labeled, ncomponents



def remove_small_components(labeled_mask,ncomponents,area):

	"""
	Removes from the mask labeled components that have less pixels than ncomponents
	"""
	sizes=[]
	labeled_mask_out = np.copy(labeled_mask)
	for label in range(ncomponents):
	    labeled_mask_aux = np.copy(labeled_mask)
	    labeled_mask_aux[labeled_mask_aux != (label+1)] = 0
	    if labeled_mask_aux.sum()/(label+1) < area:
	        labeled_mask_out[labeled_mask_out == (label+1)] = 0

	labeled_mask_out[labeled_mask_out != 0] = 1

	return labeled_mask_out


def average_intensity(masked_img, option, nb_pixels):

	"""
	Takes a dapi/uv image as input, and returns the average intensity
	Option 1: mean pixel intensity
	Option 2: median pixel intensity
	Pixel intensity computed as square_root(channel_1^2 + channel_2^2 + channel_3^2)
	Average intensity obtained by dividing total intensity by the number of pixels of interest
	"""
	
	masked_img = rgb2gray(masked_img)
	total_intensity = masked_img.sum()

	if option == 1: #mean
		intensity=total_intensity/nb_pixels
	elif option == 2: #median
		m = np.ma.masked_equal(masked_img, 0)
		intensity = np.ma.median(m)

	return intensity, total_intensity, nb_pixels


def invert_mask(mask):
	
	"""
	inverts a mask so the backgound becomes mask and the original mask becomes backgrounds.
	useful for analysis of background pixels.
	"""
	
    out = np.ones(mask.shape)
    out = out - mask
    out[out==-1] = 1
    
    return out.astype(np.int)


def apply_mask2img(img, mask):

	"""
	Applies mask to image, where pixels outside the mask are equaled to zero.
	"""
	# create a rgb image (N,M,3) by repeating the mask for each channel
	mask_rgb = np.zeros(img.shape,dtype=np.int)
	for i in range(3):
		mask_rgb[:,:,i] = mask
	return img*mask_rgb # apply mask by multlipying it with image

