import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import threshold_yen

def rgb2gray(rgb_img):
	"""
	Convert image from rgb to greyscale. not universal, specific to this problem
	"""
	gray = (0.3*rgb_img[:,:,0])+(0.59*rgb_img[:,:,1])+(0.11*rgb_img[:,:,2])

	return gray

def average_intensity(masked_img, option, nb_pixels):

	"""
	Takes a dapi image as input, and returns the average intensity
	Option 1: average pixel intensity
	Option 2: median pixel intensity
	Pixel intensity computed as square_root(channel_1^2 + channel_2^2 + channel_3^2)
	Average intensity obtained by dividing total intensity by the number of pixels of interest
	"""
	
	#nb pixels=number of pixels of interest in the mask. Because a mask is 1s and 0s, 
	#the number of pixels of interest is equal to its sum.
	masked_img = rgb2gray(masked_img)
	total_intensity = masked_img.sum()

	if option == 1: #average
		intensity=total_intensity/nb_pixels
	elif option == 2: #median - fiz um masked array para rejeitar os zeros
		m = np.ma.masked_equal(masked_img, 0)
		intensity = np.ma.median(m)

	return intensity, total_intensity, nb_pixels


def find_mask_threshold(bf_img, lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2,width_filter):

	"""
	Define mask by converting image into rgb and setting all pixels with
	values between the threshold to zero
	inside this function the background is already corrected
	"""

	bf_img_gs = rgb2gray(bf_img)
	bf_img_gs_corr=bg_correct(bf_img_gs)
	mask =  np.logical_and(bf_img_gs_corr<upper_bound_1 , bf_img_gs_corr>lower_bound_1)
	bf_img_gs_corr[mask] = 0 
	bf_img_gs_corr[bf_img_gs_corr != 0] = 1

	img1=freq_filter(bf_img_gs_corr,width_filter)
	#img_gray = np.mean(img_back, axis=1)
	mask_a =  np.logical_and(img1<upper_bound_2 , img1>lower_bound_2)
	img1[mask_a] = 1 
	img1[img1 != 1] = 0
	
	return bf_img_gs_corr.astype(np.int),img1.astype(np.int)

def dil_er(mask,kernel,it):
	"""
	Dilation followed by an erosion. Transforms the input into uint8
	Useful to disconnect holes that are have been brought together so
	the labeling and hole removal works better. 
	"""
	mask_i= mask.astype(np.uint8)
	kernel_dil = disk(kernel)
	kernel_er = disk(kernel)
	dilation = cv2.dilate(mask_i,kernel_dil,iterations = it)
	erosion = cv2.erode(dilation,kernel_er,iterations = it+1)

	return erosion

def invert_mask(mask):
    out = np.ones(mask.shape)
    out = out - mask
    out[out==-1] = 1
    
    return out.astype(np.int)

def mask_labeling(mask):
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    return labeled, ncomponents

def remove_small_components(labeled_mask,ncomponents,area):
    
	sizes=[]
	labeled_mask_out = np.copy(labeled_mask)
	for label in range(ncomponents):
	    labeled_mask_aux = np.copy(labeled_mask)
	    labeled_mask_aux[labeled_mask_aux != (label+1)] = 0
	    if labeled_mask_aux.sum()/(label+1) < area:
	        labeled_mask_out[labeled_mask_out == (label+1)] = 0

	labeled_mask_out[labeled_mask_out != 0] = 1

	return labeled_mask_out


def apply_mask2img(img, mask):

	# create a rgb image (N,M,3) by repeating the mask for each channel
	mask_rgb = np.zeros(img.shape,dtype=np.int)
	for i in range(3):
		mask_rgb[:,:,i] = mask
	return img*mask_rgb # apply mask by multlipying it with image


def eq_norm(img):
	
	#normalizes and equalized a grayscale image
	#shifts image to 0 to 1, increases contrast by setting the max to 1 and min to -1
	# eliminates outlier pixels, rescaling between 2 and 98% percentiles: increases contrast 
	
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
	img_eq_norm=img_rescale/img_rescale.max()

	return img_eq_norm


def entropy_filter (img):
	#input: normalized and equalized image
	#returns a masked image based on the entropy of the image.

	ent_img=entropy(img, disk(20))
	thresh=threshold_yen(ent_img)
	thresh_img = ent_img <= thresh
	mask = invert_mask(thresh_img.astype(int))

	return mask,ent_img


def convertImage(img1):
    
    img = img1.convert("RGBA")
  
    datas = img.getdata()
  
    newData = []
  
    for items in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
