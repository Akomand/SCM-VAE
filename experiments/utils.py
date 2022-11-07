import os
import argparse
import json
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


import sys
sys.path.append('../')
from experiments.datasets import get_synthetic_data, get_celeba_data
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def get_device():
	torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



def get_default_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_dir', type=str, required=True, default='../../data/pendulum')
	# parser.add_argument('--model_name', type=str, default='SCM_VAE')
	parser.add_argument('--iter_save', type=int, default=5, help="Save model every n epochs")
	parser.add_argument('--train', type=int, default=1, help="Flag for training")
	parser.add_argument('--data', type=str, default="pendulum", help="Flag for data")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--seed', type=int, default=42)

	return parser


def load_dataset(args):
	pl.seed_everything(args.seed)
	print('Loading datasets...')

	if args.data == 'celeba':
		data_name = 'celeba'
		train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
			get_celeba_data(args.dataset_dir, 64, type='train')
	else:
		data_name = 'pendulum'
		train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader\
			= get_synthetic_data(args.dataset_dir, 64)

		print(f'Length of training set: {len(train_dataset)}')
		print(f'Length of val set: {len(val_dataset)}')
		print(f'Length of test set: {len(test_dataset)}')
		# exit(0)



	datasets = {
		'train': train_dataset,
		'val': val_dataset,
		'test': test_dataset
	}

	dataloaders = {
		'train': train_loader,
		'val': val_loader,
		'test': test_loader
	}

	return datasets, dataloaders, data_name



CHECKPOINT_PATH = "../saved_models/scm_vae/checkpoints"
def train_model(model_class, model_name, train_loader, val_loader,
				test_loader=None,
				max_epochs=100,
				check_val_every_n_epoch=1,
				save_last_model=False,
				**kwargs):

	save_name = model_name

	trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
						 gpus=1 if str(device)=="cuda:0" else 0,
						 max_epochs=max_epochs,
						 callbacks=[ModelCheckpoint(save_weights_only=True, every_n_epochs=5),
									LearningRateMonitor("epoch")],
						 enable_progress_bar=True)

	# Path(f"{self.logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
	Path(f"{trainer.logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
	Path(f"{trainer.logger.log_dir}/Real").mkdir(exist_ok=True, parents=True)

	trainer.logger._log_graph = True
	trainer.logger._default_hp_metric = None

	# CHECK IF LOADING
	pretrained_file = os.path.join(CHECKPOINT_PATH, save_name + '.ckpt')
	if os.path.isfile(pretrained_file):
		print(f'Found pretrained model to load at {pretrained_file}, loading...')
		model = model_class.load_from_checkpoint(pretrained_file)
	else:
		pl.seed_everything(42)
		model = model_class(model_name=model_name, **kwargs)
		trainer.fit(model, train_loader, val_loader)
		model = model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # load best model


	# Test
	# trainer.test(model, test_loader, verbose=False)

	return model


def test_model(model_class, model_name, train_loader, val_loader,
				test_loader=None,
				max_epochs=100,
				check_val_every_n_epoch=1,
				save_last_model=False,
				**kwargs):
	CHECKPOINT_PATH = "../saved_models/scm_vae/checkpoints"
	# print(CHECKPOINT_PATH)
	save_name = model_name
	# print(save_name)
	# print(os.path.join(CHECKPOINT_PATH, save_name))
	trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
						 gpus=1 if str(device)=="cuda:0" else 0,
						 max_epochs=max_epochs,
						 callbacks=[ModelCheckpoint(save_weights_only=True, every_n_epochs=5),
									LearningRateMonitor("epoch")],
						 enable_progress_bar=True)

	Path(f"{trainer.logger.log_dir}/Reconstructions_Inference").mkdir(exist_ok=True, parents=True)
	Path(f"{trainer.logger.log_dir}/Interventions_Inference").mkdir(exist_ok=True, parents=True)
	Path(f"{trainer.logger.log_dir}/Real_Inference").mkdir(exist_ok=True, parents=True)

	trainer.logger._log_graph = True
	trainer.logger._default_hp_metric = None
	# print(os.getcwd())
	# CHECK IF LOADING
	# pretrained_file = CHECKPOINT_PATH + '/lightning_logs/version_13/checkpoints/epoch=99-step=6900' + '.ckpt'
	pretrained_file = '/home/akomandu/code/SCM-VAE/saved_models/scm_vae/checkpoints/SCM_VAE/lightning_logs/version_13/checkpoints/epoch=99-step=6900.ckpt'
	# pretrained_file = os.path.join(CHECKPOINT_PATH, 'lightning_logs/version_13/checkpoints/epoch=99-step=6900' + '.ckpt')
	# print(f'Found pretrained model to load at {pretrained_file}, loading...')
	# model = model_class.load_from_checkpoint(pretrained_file)
	# print(model)
	# exit(0)
	# print(pretrained_file)
	if os.path.isfile(pretrained_file):
		print(f'Found pretrained model to load at {pretrained_file}, loading...')
		model = model_class.load_from_checkpoint(pretrained_file)
	else:
		pl.seed_everything(42)
		model = model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # load best model


	# Test
	trainer.test(model, test_loader, verbose=False)

	return model