import argparse
import torch.utils.data as data
import numpy as np
import torch
import sys
import random
sys.path.append('../')
from models.scm_vae import SCM_VAE
from experiments.utils import load_dataset, get_default_parser, train_model


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/scm_vae/checkpoints"

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == '__main__':
	parser = get_default_parser()
	parser.add_argument('--z_dim', type=int, default=16)
	parser.add_argument('--z1_dim', type=int, default=4)
	parser.add_argument('--z2_dim', type=int, default=4)
	parser.add_argument('--alpha', type=float, default=0.1)
	parser.add_argument('--beta', type=float, default=1.0)
	parser.add_argument('--gamma', type=float, default=1.0)
	parser.add_argument('--lambda_v', type=float, default=1e-3)
	parser.add_argument('--inference', type=bool, default=False)
	parser.add_argument('--use_causal_prior', type=bool, default=True)
	parser.add_argument('--warmup', type=int, default=100)
	parser.add_argument('--max_iters', type=int, default=100)

	args = parser.parse_args()
	model_args = vars(args)

	datasets, data_loaders, data_name = load_dataset(args)
	model_class = SCM_VAE

	check_val_every_n_epoch = 2

	model = train_model(model_class=model_class,
						model_name="SCM_VAE",
						train_loader=data_loaders['train'],
						val_loader=data_loaders['val'],
						test_loader=data_loaders['test'],
						check_val_every_n_epoch=check_val_every_n_epoch,
						save_last_model=True,
						**model_args)
