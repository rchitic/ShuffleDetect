"""
This file acts as main file. It runs all attacks to generate the adversarial images and obtains the detection rate and false positive rate for each attack.
"""
import numpy as np
import json
import os
import sys
import time

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from torch_generate_adv import get_attack
import params
from utils import create_torchmodel_imagenet, prediction_preprocess
from shuffle_detection import *

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.deterministic = True

# Params
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
N = params.N
data_path = params.data_path
results_path = params.results_path
epsilonsFGSM = params.epsilonsFGSM
epsilons = params.epsilons
epsilons1 = params.epsilons1
epsilons2 = params.epsilons2

dataset = "imagenet"
m = []
if dataset == 'imagenet':
	networks = params.networks_imagenet
	class_dict = params.class_dict_imagenet
	names = params.names_imagenet
	for network in networks:
		m.append(create_torchmodel_imagenet(network))
	im_shape = 224

if dataset == 'cifar10':
	networks = params.networks_cifar10
	class_dict = params.class_dict_cifar10
	names = params.names_cifar10
	for network in networks:
		m.append(create_torchmodel_cifar10(network))
	im_shape = 32
	transform_test  = transforms.Compose([transforms.ToTensor()])
	testset     = torchvision.datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
	test_loader     = DataLoader(testset, batch_size=1, shuffle=False)
	# Transform images and targets from tensors to numpy arrays
	x_test = []
	y_test = []
	for idx, (data, target) in enumerate(test_loader):
		x_test.append(data.cpu().detach().numpy())
		y_test.append(target.cpu().detach().numpy())
	image_idxs = params.image_idxs

#Either specify attack from command, or run on all attacks by default
'''
atk_name = sys.argv[1]
attack_types = [atk_name]
targeted = sys.argv[2].lower() == 'true'
'''
rep = 20 # number of repetitions of the shuffle detection
label = {True:'targeted',False:'untargeted'}
labels = {'EA':[True,False],'FGSM':[True,False],'BIM':[True,False],'PGD1':[True,False],'PGD2':[True,False],'PGDInf':[True,False],'CWi':[True,False],'DF':[False]}
attack_types = ['FGSM','BIM','PGD1','PGD2','PGDInf','CWi','DF']
complete_list = []
adversarials_exist = True

for atk_name in attack_types:
	for targeted in labels[atk_name]:
		print(atk_name,targeted)
		# Select appropiate epsilons and create different dictionaries to store results
		detectinf,detect1,detect2 = {},{},{}
		fpinf,fp1,fp2 = {},{},{}
		totalinf,total1,total2 = {},{},{}
		det_all, fp_all = [],[]
		for i in epsilons:
			detectinf[i] = 0
			fpinf[i] = 0
			totalinf[i] = 0
		for i in epsilons1:
			detect1[i] = 0
			fp1[i] = 0
			total1[i] = 0
		for i in epsilons2:
			detect2[i] = 0
			fp2[i] = 0
			total2[i] = 0

		if atk_name in ['FGSM']:
			epsilon_list = epsilonsFGSM
			detect, fp, total = detectinf, fpinf, totalinf
		elif atk_name in ['BIM','PGDInf']:
			epsilon_list = epsilons
			detect, fp, total = detectinf, fpinf, totalinf
		elif atk_name in ['PGD1']:
			epsilon_list = epsilons1
			detect, fp,total = detect1, fp1, total1
		elif atk_name in ['PGD2']:
			epsilon_list = epsilons2
			detect, fp, total = detect2, fp2, total2
		else:
			epsilon_list = []
			detect, fp,total = 0, 0, 0

		# Main
		#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		for name in names:
			ca = class_dict[name][0]
			if targeted:
				ct = np.array([class_dict[name][1]])	
			else:
				ct = None

			for order in range(1,11):
				#print(f"Name {name} Order {order}")
			
				# Get ancestor image
				if dataset == 'imagenet':
					ancestor = cv2.imread(data_path.format(name,name,str(order))) #BGR image
					ancestor = cv2.resize(ancestor,(224,224))[:,:,::-1] #RGB image
					ancestor = ancestor.astype(np.uint8)
					ancestor = prediction_preprocess(Image.fromarray(ancestor)).cpu().detach().numpy()
					ancestor = ancestor.reshape(1,3,im_shape,im_shape)
				elif dataset == 'cifar10':
					ancestor = x_test[image_idxs[name][order-1]]
					ancestor = ancestor.reshape(1,3,im_shape,im_shape)

				for i,model in enumerate(m):
					network = networks[i]
					print(f"{atk_name} {network} {name} {order} {epsilon_list}")
					#print(epsilon_list)
					if len(epsilon_list)>0:
						for eps in epsilon_list:
							if atk_name in ['PGD1','PGD2']:
								adv_file_path = '{}/{}/{}_{}/{}/{}_{}_{}.npy'.format(results_path,dataset,atk_name,label[targeted],eps,network,name,order)
							else:
								adv_file_path = '{}/{}/{}_{}/{}/{}_{}_{}.npy'.format(results_path,dataset,atk_name,label[targeted],int(eps*256),network,name,order)

							if not adversarials_exist:
								# Create adversarial image
								adv = get_attack(dataset,atk_name,model,ancestor,targeted,ct,eps).reshape(3,im_shape,im_shape)
								# Save adversarial image
								np.save(adv_file_path,adv)

							if os.path.exists(adv_file_path):
								adv = np.load(adv_file_path)
								print(adv.shape)
								# Check shuffle detection (only if adversarial is successful)
								pred_adv = model(torch.from_numpy(adv).float().cuda().reshape(1,3,im_shape,im_shape)).cpu().detach().numpy()
								if (targeted and (pred_adv[0,ct] == pred_adv.max())) or (not targeted and (pred_adv[0,ca] != pred_adv.max())):
									#total[eps]+=1
									det_im, fp_im = shuffle_detect(ancestor.reshape(3,im_shape,im_shape),adv,model,rep)
									#detect += detect_im
									#fp += fp_im
									det_all.append(det_im)
									fp_all.append(fp_im)

					else:
						eps = None
						adv_file_path = '{}/{}/{}_{}/{}_{}_{}.npy'.format(results_path,dataset,atk_name,label[targeted],network,name,order)

						if not adversarials_exist:
							# Create adversarial image
							adv = get_attack(dataset,atk_name,model,ancestor,targeted,ct,eps)
							#adv = adv.reshape(3,im_shape,im_shape)
							# Save adversarial image
							np.save(adv_file_path,adv)

						if os.path.exists(adv_file_path):
							adv = np.load(adv_file_path)
							# Check shuffle detection (only if adversarial is successful)
							pred_adv = model(torch.from_numpy(adv).cuda().float().reshape(1,3,im_shape,im_shape)).cpu().detach().numpy()
							if (targeted and (pred_adv[0,ct] == pred_adv.max())) or (not targeted and (pred_adv[0,ca] != pred_adv.max())):
								#total+=1
								det_im, fp_im = shuffle_detect(ancestor.reshape(3,im_shape,im_shape),adv,model,rep)
								#detect += detect_im
								#fp += fp_im
								det_all.append(det_im)
								fp_all.append(fp_im)
						
		# Print and log results
		log_file = 'shuffle_detection_results_inceptionv3{}{}{}.log'.format(atk_name,rep,targeted)
		with open(log_file,'a') as res_file:
			res_file.write(f"Detection:\n{det_all}\nFalse positive:\n{fp_all}")


