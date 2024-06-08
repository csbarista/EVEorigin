import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet
import logging
import numpy as np
from collections import OrderedDict

from torch.utils.data import DataLoader, Subset
from datasources import EVESequences_train, EVESequences_val
from core import DefaultConfig
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime, timezone



import logging

import torch
from torch.nn import functional as F
import torchvision.utils as vutils

import core.training as training
from datasources import EVESequences_train, EVESequences_val
from models.eve import EVE


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    config, device = training.script_init_common()

    train_dataset_paths = [
                ('eve_train',
                EVESequences_train,
                '/media/linlishi/Extend/EVE/eve_dataset',
                #  '/media/linlishi/Extend/EVE/eve_dataset',
                # '/home/luanfuzi/dataset/eve_dataset',
                ['image', 'video', 'wikipedia'],
                ['basler', 'webcam_l', 'webcam_c', 'webcam_r']),  # noqa
            ]

    validation_dataset_paths = [
                ('eve_val', EVESequences_val,
                '/media/linlishi/Extend/EVE/eve_dataset',
                #  '/media/linlishi/Extend/EVE/eve_dataset',
                # '/home/luanfuzi/dataset/eve_dataset',
                ['image', 'video', 'wikipedia'],
                ['basler', 'webcam_l', 'webcam_c', 'webcam_r']),
            ]

    train_data, test_data = training.init_datasets(train_dataset_paths, validation_dataset_paths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(torch.cuda.get_device_name(device)))

    model = EVE()
    print(model)
    model = model.to(device)

    optimizers = torch.optim.Adam(
            model.parameters(),
            lr=0.008,
            weight_decay=0.005,
        )
    scheduler = ExponentialLR(optimizers, gamma=0.5)
    num_epochs = 10

    timestamp = time.time()
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc)
    writer = SummaryWriter('./tensorboard/UTC' + utc_time.strftime('%Y%m%d_%H%M%S'))

    dataloader = train_data['eve_train']['dataloader']

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            optimizers.zero_grad()
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
            output_dict = model(data)
            loss = output_dict['full_loss']
            loss.backward()
            optimizers.step()

            writer.add_scalar('full_loss', output_dict['full_loss'].item(), epoch * len(dataloader) + i)
            writer.add_scalar('loss_l1_left_pupil_size', output_dict['loss_l1_left_pupil_size'], epoch * len(dataloader) + i)
            writer.add_scalar('loss_l1_right_pupil_size', output_dict['loss_l1_right_pupil_size'], epoch * len(dataloader) + i)
            writer.add_scalar('loss_ang_left_g_initial', output_dict['loss_ang_left_g_initial'].item(), epoch * len(dataloader) + i)
            writer.add_scalar('loss_ang_right_g_initial', output_dict['loss_ang_right_g_initial'].item(), epoch * len(dataloader) + i)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{4069}], Loss: {output_dict}")
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Updated learning rate after epoch {epoch+1}: {current_lr}')
    print('Finished Training')