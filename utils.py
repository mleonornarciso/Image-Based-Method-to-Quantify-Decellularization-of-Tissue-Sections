from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil

# Load Image Pair
def load_image(file_path):
	"""
	Loads pair of images and conversts them into numpy array.

	"""
	# Load
	img_bf = Image.open(file_path+'.tif')
	img_dapi = Image.open(file_path[:-2]+'DAPI.tif')

	# Convert to numpy array
	img_bf = np.asarray(img_bf) # np acepts PIL image formats
	img_dapi = np.asarray(img_dapi)

	return [img_bf, img_dapi]

def load_image_immuno(file_path):
	"""
	Loads pair of images and conversts them into numpy array.

	"""
	# Load
	img_bf = Image.open(file_path+'.tif')
	img_dapi = Image.open(file_path[:-2]+'DAPI.tif')
	img_red = Image.open(file_path[:-2]+'TXREDe.tif')
	img_yellow = Image.open(file_path[:-2]+'FITCe.tif')

	# Convert to numpy array
	img_bf = np.asarray(img_bf) # np acepts PIL image formats
	img_dapi = np.asarray(img_dapi)
	img_red = np.asarray(img_red)
	img_yellow = np.asarray(img_yellow)

	return [img_bf, img_dapi, img_red, img_yellow]

def load_dataset(dataset_path):
	"""
	Load full dataset (given its path), and saves the data as dictionary 

	"""
	# Define dataset as dictionary, with keys corresponding to experiment names
#	dataset_dic = {
#	'Ammonia':[],
#	'CHAPS':[],
#	'control':[],
#	'SDC':[],
#	'SDS':[],
#	'trrypsds':[],
#	'trypfbs':[]
#	}
	
	dataset_dic = {}
	for exp_name in os.listdir(dataset_path):
	    dataset_dic[exp_name] = []
	for exp_dir in os.listdir(dataset_path): # iterate over experiment sub-directory
		exp_path = os.path.join(dataset_path, exp_dir)
		
		for file in os.listdir(exp_path): # iterate over files within experiment sub-directory
			filename = os.fsdecode(file)

			if filename.endswith('.tif'): # and filename[-5].isdigit():  # verify if .tif file and bf image
				"""
				heuristic to get only bf image: check if the character before the dot in the filename
				is a digit using function .isdigit()
				"""
				filename = filename[:-4] # remove ending (.tif)
				file_label = filename[-2:] # get label (PI or BF)

				if file_label == 'BF':
					# Load imagesge pair
					file_path = os.path.join(exp_path, filename)
					img_pair = load_image(file_path)

					# Save image pair to dictionary list that corresponds to experiment
					dataset_dic[exp_dir].append(img_pair)
				
	return dataset_dic

def load_dataset_immuno(dataset_path):
	"""
	Load full dataset (given its path), and saves the data as dictionary 

	"""
	# Define dataset as dictionary, with keys corresponding to experiment names
#	dataset_dic = {
#	'Ammonia':[],
#	'CHAPS':[],
#	'control':[],
#	'SDC':[],
#	'SDS':[],
#	'trrypsds':[],
#	'trypfbs':[]
#	}
	
	dataset_dic = {}
	for exp_name in os.listdir(dataset_path):
	    dataset_dic[exp_name] = []
	for exp_dir in os.listdir(dataset_path): # iterate over experiment sub-directory
		exp_path = os.path.join(dataset_path, exp_dir)
		
		for file in os.listdir(exp_path): # iterate over files within experiment sub-directory
			filename = os.fsdecode(file)

			if filename.endswith('.tif'): # and filename[-5].isdigit():  # verify if .tif file and bf image
				"""
				heuristic to get only bf image: check if the character before the dot in the filename
				is a digit using function .isdigit()
				"""
				filename = filename[:-4] # remove ending (.tif)
				file_label = filename[-2:] # get label (PI or BF)

				if file_label == 'BF':
					# Load imagesge pair
					file_path = os.path.join(exp_path, filename)
					img_pair = load_image_immuno(file_path)

					# Save image pair to dictionary list that corresponds to experiment
					dataset_dic[exp_dir].append(img_pair)
				
	return dataset_dic

def im_show_pair(img_pair):
	"""
	Displays pair of images taking list of numpy
	array as input
	"""
	fig, axs = plt.subplots(2,1) # define subplot
	for i,img in enumerate(img_pair): #iterate over subplot and image
		axs[i].imshow(img)


def im_show(img, cmap=None):
	"""
	Displays single image taking a numpy array as input
	"""
	if cmap:
		plt.imshow(img,cmap)
	else:
		plt.imshow(img)

def im_show_n(img_list):
	"""
	Display n images taking a list of numpy array
	"""

	fig, axs = plt.subplots(len(img_list),1)
	for i,img in enumerate(img_list): #iterate over subplot and image
		axs[i].imshow(img)

def im_show_n_sqr(list_img,n_col):
	n_lin = int(ceil(len(list_img)/n_col))
	fig, axs = plt.subplots(n_lin,n_col)
	
	collin = zip(range(n_lin),range(n_col))

	img_num = 0
	for i in range(n_lin): #iterate over subplot and image
		for j in range(n_col):
			axs[i,j].imshow(list_img[img_num])
			img_num += 1
