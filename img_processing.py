import numpy as np
import cv2
from scipy.ndimage.measurements import label
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import threshold_yen, threshold_otsu, threshold_mean
#from time import time

#from scipy import ndimage

def rgb2gray(rgb_img):
	"""
	Convert image from rgb to greyscale. not universal, specific to this problem
	"""
	gray = (0.3*rgb_img[:,:,0])+(0.59*rgb_img[:,:,1])+(0.11*rgb_img[:,:,2])

	return gray

def gray2rgb(gray_img):
	"""
	Convert image from grayscale to rgb 
	"""
	(N,M)=gray_img.shape
	
	out = np.zeros((N,M,3))
	out[:,:,0] = gray_img
	out[:,:,1] = gray_img
	out[:,:,2] = gray_img

	return out.astype(np.int)
	
def bg_correct (rgb_img):

	#normalize the images (switch rgb_img for img)
	#norm_img = np.zeros(img.shape)
	#rgb_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)

	n_row, n_col = rgb_img.shape
	med_row=np.zeros(n_row)
	med_col=np.zeros(n_col)
	rgb_img2=np.zeros(rgb_img.shape)

	#calcular as medianas para cada linha e coluna
	for i in range(n_row):
	    med_row[i] = np.median(rgb_img[i,:])
	for j in range(n_col):
	    med_col[j] = np.median(rgb_img[:,j])

	#expandir as dimensoes dos vetores porque senao a funcao repeat nao funciona
	med_row = np.expand_dims(med_row, axis=0)
	med_col = np.expand_dims(med_col, axis=0)

	#repetir os valores das medianas ao longo das linhas
	med_row_rp = np.repeat(med_row, n_col ,axis=0)
	med_col_rp = np.repeat(med_col, n_row ,axis=0)

	rgb_img2=(rgb_img-med_row_rp.transpose()-med_col_rp)
	rgb_img3 = rgb_img2-np.amin(rgb_img2)+1

	#for i in range(n_row):
	#    for j in range(n_col):
	#        rgb_img2[i,j] = rgb_img[i,j] -med_row[i]-med_col[j]
	return rgb_img3

def freq_filter (mask,width):
	"""
	Receive a mask and apply to it a low pass filter.
	PERCEBER MELHOR ESTA FUNÇAO.
	"""
	dft = cv2.dft(np.float32(mask),flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	mag_filtered= 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

	rows, cols = mask.shape
	crow, ccol = rows//2 , cols//2
	# create a mask first, center square is 1, remaining all zeros
	mask2 = np.zeros((rows,cols,2),np.uint8)
	mask2[crow-width:crow+width, ccol-width:ccol+width] = 1
	# apply mask and inverse DFT
	fshift = dft_shift*mask2
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

	return img_back

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
	#changed by Maria: 13/10
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
	#o astype muda de floats para 1s e zerox. e nao é so isso! muda para inteiros. que é a unica cena que importa.
	#e que faz com que possas aplicar as mascaras as imagens, disse o torron.

def closing(mask,kernel,it):
	#kernel=9,it=2
	"""
	Dilation followed by an erosion. Transforms the input into uint8
	Useful to disconnect holes that are have been brought together so
	the labeling and hole removal works better. 
	"""
	mask_i= mask.astype(np.uint8)
	kernel_dil = disk(kernel)
	kernel_er = disk(kernel)
	#dilate and then erode. Repeats each process for the number of iterations, it
	dilation = cv2.dilate(mask_i,kernel_dil,iterations = it)
	erosion = cv2.erode(dilation,kernel_er,iterations = it+1)

	return erosion

#def closing(mask,kernel,it):
	#kernel=9,it=2
	"""
	Dilation followed by an erosion. Transforms the input into uint8
	Useful to disconnect holes that are have been brought together so
	the labeling and hole removal works better. 
	"""
#	mask_i= mask.astype(np.uint8)
#	kernel_dil = np.ones((kernel,kernel),np.uint8)
#	kernel_er = np.ones((kernel,kernel),np.uint8)
	#dilate and then erode. Repeats each process for the number of iterations, it
#	dilation = cv2.dilate(mask_i,kernel_dil,iterations = it)
#	erosion = cv2.erode(dilation,kernel_er,iterations = it)

#	return erosion

def invert_mask(mask):
    out = np.ones(mask.shape)
    out = out - mask
    out[out==-1] = 1
    
    return out.astype(np.int)

def mask_labeling(mask):
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(mask, structure)
    return labeled, ncomponents

def remove_small_blobs(labeled_mask,ncomponents,area):
    
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