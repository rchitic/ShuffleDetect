from __future__ import division, absolute_import, print_function
import argparse

from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, BoundaryAttack, HopSkipJump, SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn

def get_attack(dataset,atk_name,classifier,ancestor,targeted,ct,e):
    #assert args.dataset in ['mnist', 'cifar', 'svhn', 'tiny', 'tiny_gray', 'imagenet'], \
    #    "dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    adv_path = '/home/aaldahdo/detectors/adv_data/'

    if dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as model
        model_mnist = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_mnist.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.3
        pa_th=78
        # random_restart = 20
        # x_train = model_mnist.x_train
        x_test = model_mnist.x_test
        # y_train = model_mnist.y_train
        y_test = model_mnist.y_test
        y_test_labels = model_mnist.y_test_labels
        translation = 10
        rotation = 60
    
    elif dataset == 'mnist_gray':
        from baselineCNN.cnn.cnn_mnist_gray import MNISTCNN as model
        model_mnist = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_mnist.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.3
        pa_th=78
        # random_restart = 20
        # x_train = model_mnist.x_train
        x_test = model_mnist.x_test
        # y_train = model_mnist.y_train
        y_test = model_mnist.y_test
        y_test_labels = model_mnist.y_test_labels
        translation = 10
        rotation = 60

    elif dataset == 'cifar10':
        kclassifier = PyTorchClassifier(model=classifier, loss=nn.CrossEntropyLoss(), input_shape=(3,32,32), nb_classes=10,clip_values=(0, 1))
        eps_sa=0.125
        pa_th=100
        translation = 8
        rotation = 30
    
    elif dataset == 'cifar100':
        kclassifier = PyTorchClassifier(model=classifier, loss=nn.CrossEntropyLoss(), input_shape=(3,32,32), nb_classes=100,clip_values=(0, 1))
        eps_sa=0.125
        pa_th=100
        translation = 8
        rotation = 30

    elif dataset == 'cifar_gray':
        from baselineCNN.cnn.cnn_cifar10_gray import CIFAR10CNN as model
        model_cifar = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_cifar.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_cifar.x_train
        x_test = model_cifar.x_test
        # y_train = model_cifar.y_train
        y_test = model_cifar.y_test
        y_test_labels = model_cifar.y_test_labels
        translation = 8
        rotation = 30

    elif dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as model
        model_svhn = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_svhn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_svhn.x_train
        x_test = model_svhn.x_test
        # y_train = model_svhn.y_train
        y_test = model_svhn.y_test
        y_test_labels = model_svhn.y_test_labels
        translation = 10
        rotation = 60

    elif dataset == 'svhn_gray':
        from baselineCNN.cnn.cnn_svhn_gray import SVHNCNN as model
        model_svhn = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_svhn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_svhn.x_train
        x_test = model_svhn.x_test
        # y_train = model_svhn.y_train
        y_test = model_svhn.y_test
        y_test_labels = model_svhn.y_test_labels
        translation = 10
        rotation = 60

    elif dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as model
        model_tiny = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_tiny.model
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_tiny.x_train
        #x_test = model_tiny.x_test
        # y_train = model_tiny.y_train
        #y_test = model_tiny.y_test
        #y_test_labels = model_tiny.y_test_labels
        translation = 8
        rotation = 30
        #del model_tiny

    elif dataset == 'tiny_gray':
        from baselineCNN.cnn.cnn_tiny_gray import TINYCNN as model
        model_tiny = model(mode='load', filename='cnn_{}.h5'.format(dataset))
        classifier=model_tiny.model
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        # x_train = model_tiny.x_train
        x_test = model_tiny.x_test
        # y_train = model_tiny.y_train
        y_test = model_tiny.y_test
        y_test_labels = model_tiny.y_test_labels
        translation = 8
        rotation = 30
        del model_tiny

    elif dataset == 'imagenet':
        kclassifier = PyTorchClassifier(model=classifier, loss=nn.CrossEntropyLoss(), input_shape=(224,224,3), nb_classes=1000,clip_values=(0, 1))
        eps_sa=0.125
        pa_th=100
        translation = 8
        rotation = 30

    
    #FGSM
    if atk_name == 'FGSM':
        attack = FastGradientMethod(estimator=kclassifier, eps=e, eps_step=0.01, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        return adv_data
    
    #BIM
    if atk_name == 'BIM':
        #attack = BasicIterativeMethod(estimator=kclassifier, eps=e, eps_step=0.01, max_iter=, targeted=targeted)
        attack = BasicIterativeMethod(estimator=kclassifier, eps=e, eps_step=0.5/255, max_iter=10, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        #adv_file_path = adv_path + dataset + '_bim_' + str(e) + '.npy'
        #np.save(adv_file_path, adv_data)
        #print('Done - {}'.format(adv_file_path))
        return adv_data
    
    #PGD1
    if atk_name == 'PGD1':
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=1, eps=e, eps_step=4, batch_size=1, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        #adv_file_path = adv_path + dataset + '_pgd1_' + str(e) + '.npy'
        #np.save(adv_file_path, adv_data)
        #print('Done - {}'.format(adv_file_path))
        return adv_data
    
    #PGD2
    if atk_name == 'PGD2':
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=2, eps=e, eps_step=0.1, batch_size=1, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        #adv_file_path = adv_path + dataset + '_pgd2_' + str(e) + '.npy'
        #np.save(adv_file_path, adv_data)
        #print('Done - {}'.format(adv_file_path))
        return adv_data
    
    #PGDInf
    if atk_name == 'PGDInf':
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=np.inf, eps=e, eps_step=0.01, batch_size=1, targeted=targeted)
        attack = ProjectedGradientDescent(estimator=kclassifier, eps=e, eps_step=0.5/255, max_iter=10, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        #adv_file_path = adv_path + dataset + '_pgdi_' + str(e) + '.npy'
        #np.save(adv_file_path, adv_data)
        #print('Done - {}'.format(adv_file_path))
        return adv_data

    #CWi
    if atk_name == 'CWi':
      attack = CarliniLInfMethod(classifier=kclassifier, targeted=targeted, confidence=5)
      if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
      else:
            adv_data = attack.generate(x=ancestor)
      #adv_file_path = adv_path + dataset + '_cwi.npy'
      #np.save(adv_file_path, adv_data)
      #print('Done - {}'.format(adv_file_path))
      return adv_data

    #CW2 - SLOW
    if atk_name == 'CW2':
        attack = CarliniL2Method(classifier=kclassifier, max_iter=100, batch_size=1, confidence=10, targeted=targeted)
        if targeted:
            adv_data = attack.generate(x=ancestor, y=ct)
        else:
            adv_data = attack.generate(x=ancestor)
        #adv_file_path = adv_path + dataset + '_cw2.npy'
        #np.save(adv_file_path, adv_data)
        #print('Done - {}'.format(adv_file_path))
        return adv_data

    #DF
    if atk_name == 'DF':
      attack = DeepFool(classifier=kclassifier)
      adv_data = attack.generate(x=ancestor)
      #adv_file_path = adv_path + dataset + '_df.npy'
      #np.save(adv_file_path, adv_data)
      #print('Done - {}'.format(adv_file_path))
      return adv_data

    #Spatial transofrmation attack
    if atk_name == 'STA':
      attack = SpatialTransformation(classifier=kclassifier, max_translation=translation, max_rotation=rotation)
      adv_data = attack.generate(x=ancestor)
      #adv_file_path = adv_path + dataset + '_sta.npy'
      #np.save(adv_file_path, adv_data)
      #print('Done - {}'.format(adv_file_path))
      return adv_data

    #Square Attack
    if atk_name == 'Square':
      attack = SquareAttack(estimator=kclassifier, max_iter=200, batch_size=1, eps=eps_sa)
      adv_data = attack.generate(x=ancestor)
      return adv_data

    #JSMA Attack
    if atk_name == 'JSMA':
      attack = SaliencyMapMethod(classifier=kclassifier)
      adv_data = attack.generate(x=ancestor)
      return adv_data

    #HopSkipJump Attack
    if atk_name == 'HSJ':
      attack = HopSkipJump(classifier=kclassifier, targeted=targeted, batch_size=1)
      iter_step = 10
      for i in range(4):
        if targeted:
            adv_data = attack.generate(x=ancestor, x_adv_init=adv_data, y=ct, resume=True)
            attack.max_iter = iter_step
        else:
            print(adv_data.shape)
            adv_data = attack.generate(x=ancestor, x_adv_init=adv_data, resume=True)
            attack.max_iter = iter_step
      adv_file_path = adv_path + dataset + '_hop.npy'
      #np.save(adv_file_path, adv_data)
      print('Done - {}'.format(adv_file_path))
      return adv_data

    #ZOO attack
    if atk_name == 'Zoo':
      attack = ZooAttack(classifier=kclassifier)
      adv_data = attack.generate(x=ancestor)
      adv_file_path = adv_path + dataset + '_zoo.npy'
      #np.save(adv_file_path, adv_data)
      print('Done - {}'.format(adv_file_path))
      return adv_data

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn', or 'tiny'",
        required=True, type=str
    )
    parser.add_argument(
        '-i', '--batch_indx',
        help="it is used if you need to generate specific AEs to start with batch indx and to end after one batch only",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="it is used if you need to generate specific AEs to start with batch indx and to end after one batch only",
        required=False, type=int
    )
    parser.add_argument(
        '-g', '--gpu',
        help="GPU Support",
        required=False, type=bool
    )
    parser.set_defaults(gpu=False)
    parser.set_defaults(batch_size=2)
    parser.set_defaults(batch_indx=0)
    args = parser.parse_args()
    '''
    #main(args)