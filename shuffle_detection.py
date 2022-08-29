'''
Shuffle adversarial images with shuffle sizes: 16,32,56,112.
Check how the CNNs classify the shuffled adverarial images.
'''

# general
import os
import time
import random
from random import shuffle
import math
import sys
import numpy as np
import itertools

# image loading
from PIL import Image
import cv2

# torch
import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#print(torch.rand(1, device="cuda"))
#torch.cuda.empty_cache()
from torchvision import transforms

#own
from utils import softmax
import params

# gpu
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

np.random.seed(0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_network(model, images):
    with torch.no_grad():
        images_copy = images.copy()
        preprocessed_images = torch.from_numpy(images_copy.reshape(1,images_copy.shape[0],images_copy.shape[1],images_copy.shape[2])).type(torch.FloatTensor).to(device)
        preds = model(preprocessed_images).cpu().detach().numpy()
    preds_softmax = softmax(preds) 
    return preds_softmax

def create_comb(shuffle_size,half):
    comb1=np.array(list(itertools.product(range(half,224,shuffle_size),range(half,224,shuffle_size))))
    comb2=np.array(list(itertools.product(range(half,224,shuffle_size),range(half,224,shuffle_size))))    
    np.random.shuffle(comb1)
    np.random.shuffle(comb2)
    return comb1, comb2

def shuffle(im, comb1, comb2, half):
    locs = []
    shuffled_im = np.zeros((im.shape[0],im.shape[1],im.shape[2]))
    for x in range(len(comb1)): 
        bbox_centre1_ij = comb1[x]
        bbox_centre2_ij = comb2[x]
        loc_i = bbox_centre1_ij[0]
        loc_j = bbox_centre1_ij[1]
        loc_i2 = bbox_centre2_ij[0]
        loc_j2 = bbox_centre2_ij[1]
        shuffled_im[:,loc_i2-half:loc_i2+half,loc_j2-half:loc_j2+half] = im[:,loc_i-half:loc_i+half,loc_j-half:loc_j+half]
    return shuffled_im

# params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
combs1, combs2 = [], []
#threshold = 10

# Main
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def shuffle_detect(ancestor,adv,model,rep):
	pred_unshuffled_ancestor = np.argmax(run_network(model,ancestor))
	pred_unshuffled_adv = np.argmax(run_network(model,adv))

	if adv.shape[1] == 224:
		shuffle_size = params.shuffle_size_imagenet
		if int(rep)==1:
			shuffle_combs_path = params.shuffle_combs_path_imagenet_single
		else:
			shuffle_combs_path = params.shuffle_combs_path_imagenet_multiple

	elif adv.shape[1] == 32:
		shuffle_size = params.shuffle_size_cifar10
		shuffle_combs_path = params.shuffle_combs_path_cifar10

	half = int(shuffle_size/2)
	if int(rep)==1:
		# below code runs one single permutation to shuffle the images
		comb1 = np.load(shuffle_combs_path+str(shuffle_size)+'/comb1.npy')
		comb2 = np.load(shuffle_combs_path+str(shuffle_size)+'/comb2.npy')

		# Shuffle images
		ancestor = shuffle(ancestor,comb1,comb2,half)
		adv = shuffle(adv,comb1,comb2,half)
	
		# predict using pre-trained network
		pred_shuffled_ancestor = np.argmax(run_network(model,ancestor))
		pred_shuffled_adv = np.argmax(run_network(model,adv))
	
		# check if label is the sam before and after shuffling
		detect = pred_unshuffled_adv != pred_shuffled_adv
		fp = pred_unshuffled_ancestor != pred_shuffled_ancestor
		
		return detect, fp

	else:
		detect, fp = 0,0
		combs1 = np.load(shuffle_combs_path+str(shuffle_size)+'/combs_{}_1.npy'.format(rep))
		combs2 = np.load(shuffle_combs_path+str(shuffle_size)+'/combs_{}_2.npy'.format(rep))

		for rep in range(len(combs1)):
			comb1, comb2 = combs1[rep], combs2[rep]
			# Shuffle images
			ancestor = shuffle(ancestor,comb1,comb2,half)
			adv = shuffle(adv,comb1,comb2,half)

			# predict using pre-trained network
			pred_shuffled_ancestor = np.argmax(run_network(model,ancestor))
			pred_shuffled_adv = np.argmax(run_network(model,adv))

			# check if label is the same before and after shuffling
			detect += pred_unshuffled_adv != pred_shuffled_adv
			fp += pred_unshuffled_ancestor != pred_shuffled_ancestor

		# The above code gets the sum of detected & false positives from the total number of runs, 
		# for example 13 detected and 4 false positives in 20 runs.
		# The code below uses a threshold to obtain a final 0 or 1 value for the detection and for the false positives,
		# for example if the threshold is 10: 13, 4 -> 1, 0
		'''
		if detect > threshold:
			detect_final = 1
		else:
			detect_final = 0
		if fp > threshold:
			fp_final = 1
		else:
			fp_final = 0
		'''

		return detect, fp # or return detect_final, fp_final if threshold is set


