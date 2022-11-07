import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
import os
from collections import OrderedDict, defaultdict
import models.shared.utils as ut
import sys
sys.path.append('../')
import random
import torchvision.utils as vutils

# model layers imports
from models.shared import ConvEncoder, ConvDecoder, ConvDec, Encoder, Decoder_DAG, DagLayer, MaskLayer, CosineWarmupScheduler

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# LR SCHEDULER NOT NEEDED - CAN SKEW RESULTS

class SCM_VAE(pl.LightningModule):
	"""The main module implementing SCM-VAE"""

	def __init__(self, model_name, z_dim, z1_dim, z2_dim, alpha=0.3, beta=1, gamma=1, lambda_v=0.0001, inference=False, use_causal_prior=True, warmup=100, max_iters=100000, **kwargs):
		super().__init__()
		self.z2_dim = z2_dim
		self.save_hyperparameters()
		self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])

		self.encoder = Encoder(self.hparams.z_dim)
		self.decoder = Decoder_DAG(self.hparams.z_dim, self.hparams.z1_dim, self.hparams.z2_dim)

		self.dag = DagLayer(self.hparams.z1_dim, self.hparams.z1_dim, i = self.hparams.inference)

		self.mask_z = MaskLayer(self.hparams.z_dim)


	def forward(self, x, label, mask=None, value=None):
		z_sample, z_masked, z_m, z_v = self.encode(x, label=label, mask=mask, value=value)

		x_hat, _, _, _, _ = self.decoder.decode_sep(
			z_sample.reshape([z_sample.size()[0], self.hparams.z_dim]), label.to(device))

		return x_hat, z_sample, z_masked, z_m, z_v, label


	def encode(self, x, label=None, mask=None, value=None):
		z_m, z_v = self.encoder.encode(x)
		z_m = z_m.reshape([z_m.size()[0], self.hparams.z1_dim, self.hparams.z2_dim])  # RESHAPE TO (BATCH, 4, 4)
		z_v = z_v.reshape([z_m.size()[0], self.hparams.z1_dim, self.hparams.z2_dim])  # RESHAPE TO (BATCH, 4, 4)

		z_temp = torch.clone(z_m)
		# NO DAG LEARNING SO GO STRAIGHT TO MASKING
		for i in range(4):
			if i == 0 or i == 1:
				z_temp[:, i, :] = z_m[:, i, :]
			else:
				z_temp[:, i, :] = self.mask_z.g(
					self.dag.mask_z(z_temp, i).reshape([z_m.size()[0], self.hparams.z_dim])).reshape(
					[z_m.size()[0], self.hparams.z2_dim]).to(device)

			if mask == i:
				z_temp[:, mask, :], z_v[:, mask, :] = self.perform_intervention(z_temp, z_v, mask, label, value)

		z_masked = z_temp

		# FINAL "CAUSAL" REPRESENTATION
		z_sample = ut.conditional_sample_gaussian(z_masked, z_v * self.hparams.lambda_v)

		return z_sample, z_masked, z_m, z_v


	def perform_intervention(self, z_temp, z_v, mask, label, value):
		label[:, mask] = value
		cp_m, cp_v = ut.condition_prior(self.scale, label, self.hparams.z2_dim)
		z_temp[:, mask, :] = cp_m[:, mask, :].to(device)
		z_v[:, mask, :] = torch.abs(cp_v[:, mask, :])

		return z_temp[:, mask, :], z_v[:, mask, :]


	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))

		# lr_scheduler = CosineWarmupScheduler(optimizer,
		# 									 warmup=[200*self.hparams.warmup, 2*self.hparams.warmup, 2*self.hparams.warmup],
		# 									 offset=[10000, 0, 0],
		# 									 max_iters=self.hparams.max_iters)

		return [optimizer]#, [{'scheduler': lr_scheduler, 'interval': 'step'}]


	def _get_loss(self, x, label, mode="train"):
		"""Calculate loss"""

		x_hat, z_sample, z_masked, z_m, z_v, label = self.forward(x, label)

		# RECONSTRUCTION LOSS
		rec = ut.log_bernoulli_with_logits(x, x_hat.reshape(x.size()))
		rec = -torch.mean(rec)

		# print(rec)

		# PRIORS
		p_m, p_v = torch.zeros(z_m.size()), torch.ones(z_m.size())
		cp_m, cp_v = ut.structural_condition_prior(self.scale, label, self.hparams.z2_dim, self.dag.A)

		cp_v = torch.ones([z_m.size()[0], self.hparams.z1_dim, self.hparams.z2_dim]).to(device)
		cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))

		# KL-DIVERGENCE BETWEEN DISTRIBUTION FROM ENCODER AND THE ISOTROPIC GAUSSIAN PRIOR
		kl = torch.zeros(1).to(device)

		# RESHAPE
		z_m = z_m.view(-1, self.hparams.z_dim).to(device)
		z_v = z_v.view(-1, self.hparams.z_dim).to(device)
		p_m = p_m.view(-1, self.hparams.z_dim).to(device)
		p_v = p_v.view(-1, self.hparams.z_dim).to(device)

		kl = self.hparams.alpha * ut.kl_normal(z_m, z_v, p_m, p_v)

		for i in range(self.hparams.z1_dim):
			kl = kl + self.hparams.beta * ut.kl_normal(z_masked[:, i, :].to(device), cp_v[:, i, :].to(device),
									   cp_m[:, i, :].to(device), cp_v[:, i, :].to(device))

		kl = torch.mean(kl)

		neg_elbo = rec + kl

		# Logging
		self.log(f'{mode}_kld', kl)
		self.log(f'{mode}_rec_loss_t1', rec)
		self.log(f'{mode}_neg_elbo', neg_elbo)

		return neg_elbo, kl, rec, x_hat.reshape(x.size()), z_sample, cp_m



	def training_step(self, batch, batch_idx):
		X, label = batch
		neg_elbo, kl, rec, x_hat, z_sample, cp_m = self._get_loss(X, label, mode="train")

		self.log('neg_ELBO', neg_elbo)
		self.log('kld', kl)
		self.log('reconstruction_err', rec)

		values = {
			'loss': neg_elbo,
			'kl': kl,
			'rec': rec,
			'x_hat': x_hat,
			'z_sample': z_sample,
			'cp_m': cp_m,
			'x': X
		}

		return values

	def validation_step(self, batch, batch_idx):
		X, label = batch
		neg_elbo, kl, rec, x_hat, z_sample, cp_m = self._get_loss(X, label, mode="val")

		self.log('neg_ELBO', neg_elbo)
		self.log('kld', kl)
		self.log('reconstruction_err', rec)


	def test_step(self, batch, batch_idx, mask=0, value=0):
		X, label = batch
		# print(X.shape)
		x_hat, z_sample, z_masked, z_m, z_v, label = self.forward(X, label, mask=mask, value=value)
		x_hat = x_hat.reshape(-1, 4, 96, 96)

		vutils.save_image(X,
						  os.path.join(self.logger.log_dir,
									   "Real_Inference",
									   f"real_{self.logger.name}_Epoch_{self.current_epoch}.png"),
						  normalize=True,
						  nrow=12)

		if mask == None:
			vutils.save_image(x_hat.data,
							  os.path.join(self.logger.log_dir,
										   "Reconstructions_Inference",
										   f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
							  range=(0, 1),
							  nrow=12)
		else:
			# print(x_hat.shape)
			vutils.save_image(x_hat.data,
							  os.path.join(self.logger.log_dir,
										   "Interventions_Inference",
										   f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
							  range=(0, 1),
							  nrow=12)


	def training_epoch_end(self, outputs):
		choice = random.choice(outputs)
		data = choice['x']
		output_sample = choice['x_hat']
		# output_sample = output_sample.reshape(-1, 3, 128, 128)

		vutils.save_image(data,
						  os.path.join(self.logger.log_dir,
									   "Real",
									   f"real_{self.logger.name}_Epoch_{self.current_epoch}.png"),
						  normalize=True,
						  nrow=12)
		vutils.save_image(output_sample.data,
						  os.path.join(self.logger.log_dir,
									   "Reconstructions",
									   f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
						  range=(0,1),
						  nrow=12)

	@staticmethod
	def get_callbacks(exmp_inputs=None, dataset=None, **kwargs):
		lr_callback = LearningRateMonitor('step')
		return [lr_callback]





