# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class dataload(data.Dataset):
	def __init__(self, root):
		imgs = os.listdir(root)
		self.imgs = [os.path.join(root, k) for k in imgs]
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		img_path = self.imgs[idx]
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)
		return data

	def __len__(self):
		return len(self.imgs)



class CelebA(Dataset):
	def __init__(self, root):
		# root = root + "/" + dataset

		imgs = os.listdir(root)
		imgs = sorted(imgs)

		# self.dataset = dataset
		self.imgs = [os.path.join(root, k) for k in imgs]

		# gender, smile, eyes open, mouth open
		# self.imglabel = np.loadtxt('./causal_data/causal_data/celeba/list_attr_celeba.txt', usecols=(21, 32, 24, 22))

		# age, gender, bald, beard
		self.imglabel = np.loadtxt('./causal_data/causal_data/celeba/list_attr_celeba.txt', usecols=(40, 21, 5, 25))

		self.transforms = transforms.Compose([
			transforms.CenterCrop(128),
			transforms.Resize((128, 128)),
			transforms.ToTensor()
		])

	def __getitem__(self, idx):
		# print(idx)

		# INDEX OF THE IMAGE IN DIRECTORY
		img_path = self.imgs[idx]

		# LABEL OF THE IMAGE
		label = torch.from_numpy(np.asarray(self.imglabel[idx]))
		# print(len(label))

		# CONVERT IMAGE TO NUMPY ARRAY
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		array1 = np.asarray(label)

		# CONVERT DATA AND LABEL TO TENSORS
		label = torch.from_numpy(array1)
		data = torch.from_numpy(array)

		# APPLY TRANSFORMS IF ANY
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)

		# RETURN (IMAGE, LABEL) PAIR
		return data, label.float()

	def __len__(self):
		return len(self.imgs)


class Synthetic(data.Dataset):
	def __init__(self, root, dataset="train"):
		root = root + "/" + dataset

		imgs = os.listdir(root)

		self.dataset = dataset

		self.imgs = [os.path.join(root, k) for k in imgs]
		self.imglabel = [list(map(int, k[:-4].split("_")[1:])) for k in imgs]
		# print(self.imglabel)
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		# print(idx)
		img_path = self.imgs[idx]

		label = torch.from_numpy(np.asarray(self.imglabel[idx]))
		# print(len(label))
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		array1 = np.asarray(label)
		label = torch.from_numpy(array1)
		data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)

		return data, label.float()

	def __len__(self):
		return len(self.imgs)


def get_synthetic_data(dataset_dir, batch_size, dataset="train"):
	dataset = Synthetic(dataset_dir, "train")

	train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.8))))
	val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.8), len(dataset))))
	test_dataset = Synthetic(dataset_dir, "test")

	train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_celeba_data(dataset_dir, batch_size, type="train", shuf=False):
	n = 20000
	dataset = CelebA(dataset_dir)

	train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.7))))
	val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85))))
	test_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.85), len(dataset))))

	train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
