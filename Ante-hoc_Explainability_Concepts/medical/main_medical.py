# -*- coding: utf-8 -*-
""" Code for training and evaluating A Framework for Learning Ante-hoc Explainable Models via Concepts.
Copyright (C) 2022 Anirban Sarkar <anirbans@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Standard Imports
import sys
import os
import numpy as np
import pdb
import pickle
import argparse
import operator
import matplotlib
# import matplotlib.pyplot as plt
from PIL import Image

# Torch Imports
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader

# Local imports
from SENN.arglist import get_senn_parser  # parse_args as parse_senn_args
from SENN.models import GSENN
from SENN.conceptizers import image_fcc_conceptizer, image_cnn_conceptizer, input_conceptizer, image_resnet_conceptizer, EfficientNet, DenseNet
# from SENN.conceptizers import *

from SENN.parametrizers import image_parametrizer, torchvision_parametrizer, vgg_parametrizer
from SENN.aggregators import linear_scalar_aggregator, additive_scalar_aggregator
from SENN.trainers import HLearningClassTrainer, VanillaClassTrainer, GradPenaltyTrainer
from SENN.utils import plot_theta_stability, generate_dir_names, noise_stability_plots, concept_grid
from SENN.eval_utils import estimate_dataset_lipschitz
from datasets.choose_dataset import select_dataset
# from prettytable import PrettyTable

# TODO: GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def load_medical_data(args, valid_size=0.1, shuffle=True, resize=None, random_seed=2008, batch_size=64, num_workers=0):
    """
        We return train and test for plots and post-training experiments

        Return:
            train_loader, valid_loader, test_loader, train, test
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # normalized according to pytorch torchvision guidelines https://chsasank.github.io/vision/models.html
    # train = CIFAR10('../../../datasets/Cifar-10/', train=True, download=False, transform=transform_train)
    # test  = CIFAR10('../../../datasets/Cifar-10/', train=False, download=False, transform=transform_test)
    train, test = select_dataset(args)  # TODO: SkinCon only at this time

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)  # TODO
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test


def parse_args():
    senn_parser = get_senn_parser()

    # Local ones
    parser = argparse.ArgumentParser(parents=[senn_parser], add_help=False,
                                     description='Interpteratbility robustness evaluation on MNIST')

    # setup
    parser.add_argument('-d', '--datasets', nargs='+',
                        default=['heart', 'ionosphere', 'breast-cancer', 'wine', 'heart',
                        'glass', 'diabetes', 'yeast', 'leukemia', 'abalone'], help='<Required> Set flag')
    parser.add_argument('--lip_calls', type=int, default=10,
                        help='ncalls for bayes opt gp method in Lipschitz estimation')
    parser.add_argument('--lip_eps', type=float, default=0.01,
                        help='eps for Lipschitz estimation')
    parser.add_argument('--lip_points', type=int, default=100,
                        help='sample size for dataset Lipschitz estimation')
    parser.add_argument('--optim', type=str, default='gp',
                        help='black-box optimization method')

    args = parser.parse_args()

    # print arguments
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print("Total Trainable Params: {}".format(total_params))
    return total_params


def main():

    # get args
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.nclasses = args.nclasses
    args.theta_dim = args.nclasses
    H, W = 224, 224
    args.input_dim = H * W

    # generate dir
    model_path, log_path, results_path = generate_dir_names(args.dataset, args)

    # dataloader
    train_loader, valid_loader, test_loader, train_tds, test_tds = load_medical_data(
                        args, batch_size=args.batch_size, num_workers=args.num_workers,
                        resize=(H, W))

    # data description: number of each label
    # fixme: class-imbalance
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for label in train_loader.dataset.labels:
        if label == 0:
            count_0 += 1
        elif label == 1:
            count_1 += 1
        else:
            count_2 += 1
    print("num of label 0: ", count_0)  # 2308
    print("num of label 1: ", count_1)  # 431
    print("num of label 2: ", count_2)  # 479

    # conceptizer of SENN
    if args.h_type == 'input':
        conceptizer = input_conceptizer()
        args.nconcepts = args.input_dim + int(not args.nobias)
    elif args.h_type == 'cnn':  # default choice
        conceptizer = image_resnet_conceptizer(args.input_dim, args.nconcepts, args.nclasses, args.concept_dim, nchannel=3)
    else:
        conceptizer = image_fcc_conceptizer(args.input_dim, args.nconcepts, args.concept_dim, nchannel=3)

    # parametrizer of SENN
    if args.theta_arch == 'simple':
        parametrizer = image_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, nchannel=3, only_positive=args.positive_theta)
    elif 'vgg' in args.theta_arch:
        parametrizer = vgg_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch=args.theta_arch, nchannel=3, only_positive=args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)
    else:
        parametrizer = torchvision_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch = args.theta_arch, nchannel=3, only_positive = args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)

    # aggregator of SENN
    aggregator = additive_scalar_aggregator(args.nconcepts, args.concept_dim, args.nclasses)

    # SENN model
    # model = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)
    model = GSENN(conceptizer, aggregator)

    # trainer
    if args.theta_reg_type in ['unreg', 'none', None]:
        trainer = VanillaClassTrainer(model, args)
    elif args.theta_reg_type == 'grad1':
        trainer = GradPenaltyTrainer(model, args, typ=1)
    elif args.theta_reg_type == 'grad2':
        trainer = GradPenaltyTrainer(model, args, typ=2)
    elif args.theta_reg_type == 'grad3':
        trainer = GradPenaltyTrainer(model, args, typ=3)
    elif args.theta_reg_type == 'crosslip':
        trainer = CLPenaltyTrainer(model, args)
    else:
        raise ValueError('Unrecognized theta_reg_type')

    # train
    if args.train or not args.load_model or (not os.path.isfile(os.path.join(model_path, 'model_best.pth.tar'))):
        trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
        trainer.plot_losses(save_path=results_path)
    else:
        checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'), map_location=lambda storage, loc: storage)
        checkpoint.keys()
        model = checkpoint['model']
        trainer = VanillaClassTrainer(model, args)  # arbitrary trained, only need to compute val acc

    model.eval()

    # Check accuracy with the best model
    checkpoint = torch.load(os.path.join(model_path, 'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    checkpoint.keys()
    model = checkpoint['model']
    trainer = VanillaClassTrainer(model, args)
    trainer.evaluate(test_loader, fold='test')

    # count num of parameters
    # # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print (pytorch_total_params)
    # count_parameters(model)

    # All_Results = {}

    # 0. Concept Grid for Visualization
    concept_grid(args, model, test_loader, top_k=10, cuda=args.cuda, save_path=results_path + '/concept_grid.pdf')


if __name__ == '__main__':
    main()
