import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary as infosummary
import contextlib

from config import *
from utils.utils import get_images_and_labels, load_state, set_freeze_root_children, change_learning_rate
from datasets import ClassificationDataset
from trainers import ClassifcationTrainer
from models.vit import ViT
from models.sl_vit import SL_ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'


augments = A.Compose([
    A.D4(),
    A.AutoContrast(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NORM_MEANS, std=IMAGE_NORM_STD)
])



def main():
    # 1. Train teacher model from scratch
    model = SL_ViT(
        in_channels=IN_CHANNELS,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        shift_ratio=int(0.5 * PATCH_SIZE),
        image_size=IMAGE_SIZE,
        mlp_dim=MLP_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    infosummary(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE), col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"])

    # Configs loaded from config.py
    train_test_model(model, checkpoint_path=f'checkpoints/{model.__class__.__name__}_D-{D_MODEL}_H-{NUM_HEADS}_L-{NUM_LAYERS}_I{IMAGE_SIZE}P{PATCH_SIZE}_MLP-{MLP_DIM}.pt')



def train_test_model(model: nn.Module, checkpoint_path: str):
    train_path = f'{DB_PATH}/train/'
    train_images, train_labels, val_images, val_labels = get_images_and_labels(train_path, limit_per_class=IMAGE_LIMIT_PER_CLASS, val_split=0.1, shuffle_seed=123, print_info=True)
    
    train_db = ClassificationDataset(train_images, train_labels, IMAGE_SIZE, db_path_root=train_path, augments=augments, transforms=_transforms)
    val_db = ClassificationDataset(val_images, val_labels, IMAGE_SIZE, augments=None, transforms=_transforms)

    
    optim = torch.optim.AdamW(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = ClassifcationTrainer(model, optim, loss_fn, NUM_CLASSES, device=device)
    trainer.fit(NUM_EPOCHS, train_db, val_db, BATCH_SIZE, checkpoint_path=checkpoint_path)

    print('*'*30)
    print('Testing')
    print('*'*30)
    test_path = f'{DB_PATH}/test/'
    test_images, test_labels, _, _ = get_images_and_labels(test_path, limit_per_class=100000, val_split=0.0, print_info=True)

    test_db = ClassificationDataset(test_images, test_labels, IMAGE_SIZE, augments=None, transforms=_transforms)
    test_loader = DataLoader(test_db, batch_size=32, shuffle=False, num_workers=2)
    model, optim = load_state(checkpoint_path, model, optim)
    trainer.evaluate(test_loader, conf=False)


if __name__ == '__main__':
    main()