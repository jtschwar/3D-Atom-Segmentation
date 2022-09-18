import torch.nn as nn
import torch
import random

from config import load_config
from utils import get_logger

import datasets.utils as ds_utils
from models import UNet3D
import metrics, losses
import trainer
import utils 

logger = get_logger('TrainingSetup')

# Load and log experiment configuration
config = load_config()
logger.info(config)

manual_seed = config.get('manual_seed', None)
if manual_seed is not None:
    logger.info(f'Seed the RNG for all devices with {manual_seed}')
    logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

# Create the model
if config['model']['name'] == 'UNet3D': 
    model = UNet3D.UNet3D(config['model'])

# use DataParallel if more than 1 GPU available
device = config['device']
if torch.cuda.device_count() > 1 and not device.type == 'cpu':
    model = nn.DataParallel(model)
    logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

# Move model into GPUs
logger.info(f"Sending the model to '{config['device']}'")
model = model.to(device)

# Report the number of learnable parameters in model
logger.info(f'Number of learnable params {utils.get_number_of_learnable_parameters(model)}')

# Create loss criterion
loss_criterion = losses.get_loss_criterion(config)

# Create evaluation metric
eval_criterion = metrics.get_evaluation_metric(config)

# Create data loaders
loaders = ds_utils.get_train_loaders(config)

# Create the optimizer
optimizer = utils.create_optimizer(config['optimizer'], model)

# Create learning rate adjustment strategy
lr_scheduler = utils.create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

# Create tensorboard formatter
trainer_config = config['trainer']
tensorboard_formatter = utils.get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))

# Create trainer
resume = trainer_config.pop('resume', None)

trainer = trainer.UNet3DTrainer(model=model,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                loss_criterion=loss_criterion,
                                eval_criterion=eval_criterion,
                                tensorboard_formatter=tensorboard_formatter,
                                device=config['device'],
                                loaders=loaders,
                                resume=resume,
                                **trainer_config)

# Start training
trainer.fit()

