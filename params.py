networks_imagenet = ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet']
networks_cifar10 = ['ResNet50']

class_dict_imagenet = {'abacus':[398,421],'acorn':[988,306],'baseball':[429,618],'brown_bear':[294,724],'broom':[462,273],'canoe':[472,176],'hippopotamus':[344,927],'llama':[355,42],'maraca':[641,112],'mountain_bike':[671,828]}
targets = [bannister,rhinoceros beetle, ladle, dingo, pirate, saluki, trifle, agama, conch, strainer]
class_dict_cifar10 = {'airplane':[0,1],'automobile':[1,2],'bird':[2,3],'cat':[3,4],'deer':[4,5],'dog':[5,6],'frog':[6,7],'horse':[7,8],'ship':[8,9],'truck':[9,0]}

names_imagenet = list(class_dict_imagenet.keys())
names_cifar10 = list(class_dict_cifar10.keys())

image_idxs = {
	'airplane':	[814,	6355,	1926,	7900,	90,		7871,	9248,	1161,	6508,	1748	],
	'automobile':	[9824,	8495,	4129,	4372,	6159,	8482,	440,	3246,	3039,	4320	],
	'bird':		[67,		3728,	3401,	1623,	35,		3947,	7845,	3322,	7508,	4208	],	
	'cat':		[302,	7212,	1124,	8594,	3463,	7210,	1861,	4637,	8141,	2207	],
	'deer':		[455,	5059,	8926,	1431,	5983,	5520,	1544,	8322,	9895,	4731	],
	'dog':		[9922,	3379,	4406,	768,	2384,	5222,	4623,	4626,	6226,	1818	],
	'frog':		[392,	7679,	7357,	9195,	3124,	6392,	8544,	7969,	7117,	5159	],
	'horse':	[17,		4905,	194,	3689,	9331,	4272,	1494,	9186,	3470,	5981	],
	'ship':		[8968,	8278,	1294,	6779,	8905,	1713,	1011,	1253,	3155,	480		],
	'truck':	[9594,	8742,	8871,	2754,	9312,	8877,	9667,	9324,	1668,	7640	]
}

shuffle_size_imagenet = 56
shuffle_size_cifar10 = 16

rgb = {0:'R',1:'G',2:'B'}

# EA
pop_size = 40
G = 103000

# white-box
N = 5
#epsilons=[  3/256,   4/256,   5/256,   6/256,   7/256,   8/256,   9/256,  10/256,  16/256,  32/256,  64/256,  80/256, 128/256]
epsilons=[1/255]
epsilonsFGSM=[2/255]
epsilons1=[5, 10, 15, 20, 25, 30, 40]
epsilons1=[30]
epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
epsilons2=[1]

data_path = '/home/users/rchitic/ShuffleDet/data/{}/{}{}.JPEG'
patch_replacement_combs_path = '/home/users/rchitic/ShuffleDet/data/patch_replacement_combs/patch_size{}.npy'
results_path = '/home/users/rchitic/ShuffleDet/results'
analysis_results_path = '/home/users/rchitic/ShuffleDet/analysis_results'

shuffle_combs_path_imagenet_single = '/home/users/rchitic/ShuffleDet/data/shuffle_combs/imagenet/single/'
shuffle_combs_path_imagenet_multiple = '/home/users/rchitic/ShuffleDet/data/shuffle_combs/imagenet/multiple/'
shuffle_combs_path_cifar10 = '/home/users/rchitic/ShuffleDet/data/shuffle_combs/cifar10/'
