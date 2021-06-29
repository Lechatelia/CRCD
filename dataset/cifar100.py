from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if hasattr(self, 'train_data'):
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.data[index], self.targets[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

def get_cifar100_dataloaders_two(opt, batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    if opt.is_autoaugment:
        # the policy is the same as CIFAR10
        train_transform.transforms.append(CIFAR10Policy())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                           std=(0.2673, 0.2564, 0.2762)))

    if opt.is_cutout:
        # use random erasing to mimic cutout
        try:
            train_transform.transforms.append(transforms.RandomErasing(p=opt.erase_p,
                                                                       scale=(0.0625, 0.1),
                                                                       ratio=(0.99, 1.0),
                                                                       value=0, inplace=False))
        except:
            print('torch RandomErasing is not available, so using own version! ')
            from .RandomReasing import RandomErasing
            train_transform.transforms.append(RandomErasing(p=opt.erase_p,
                                                            scale=(0.0625, 0.1),
                                                            ratio=(0.99, 1.0),
                                                            value=0, inplace=False))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=TwoCropsTransform(train_transform))
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=TwoCropsTransform(train_transform))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k # number of negative samples for NCE
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if hasattr(self, 'train_data'):
            self.old_cifar = True
        else:
            self.old_cifar = False
        if self.old_cifar:
            if self.train:
                num_samples = len(self.train_data) 
                label = self.train_labels
            else:
                num_samples = len(self.test_data)
                label = self.test_labels
        else:

            num_samples = len(self.data) 
            label = self.targets


        self.cls_positive = [[] for i in range(num_classes)] #
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i) 

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j]) 

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)#
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) # 100, 500
        self.cls_negative = np.asarray(self.cls_negative) #　100, 49500
        if self.mode== "queue":
            import queue
            # sample_idx = np.random.choice(np.arange(num_samples), self.k + 64, replace=False) # 
            # self.sample_idx_queue = queue.Queue(maxsize=self.k + 64)
            # for i in sample_idx:
            #     self.sample_idx_queue.put(i)
            self.bs = 64
            self.sample_idx = np.random.choice(np.random.choice(np.arange(num_samples), 10000, replace=False),[self.k + 64], replace=True)
            self.ptr1 = 0
            self.ptr2 = self.k
            print("the len of queue is {}".format(len(list(self.sample_idx))))



    def __getitem__(self, index):
        index_tem = index
        if self.old_cifar:

            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1) # choose a positive
                pos_idx = pos_idx[0]
            elif self.mode == 'queue':
                pos_idx = [index] # choose a positive
                if self.ptr2 > self.ptr1:
                    neg_idx = self.sample_idx[self.ptr1:self.ptr2].copy()
                else:
                    neg_idx = np.concatenate((self.sample_idx[:self.ptr2], self.sample_idx[self.ptr1:])).copy()
                self.sample_idx[self.ptr1] = index
                assert neg_idx.shape[0] == self.k
                self.ptr1 += 1
                self.ptr2 += 1
                if self.ptr1 >= self.k + self.bs:
                    self.ptr1 = 0
                if self.ptr2 > self.k + self.bs:
                    self.ptr2 = 1
                sample_idx = np.hstack((np.asarray(pos_idx), neg_idx))
                # if self.sample_idx_queue.full():
                #     self.sample_idx_queue.get()
                # self.sample_idx_queue.put(index, False)
                return img, target, index, sample_idx
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2), # 
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
