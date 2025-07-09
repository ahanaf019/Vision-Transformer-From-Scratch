import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from torchmetrics import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter

from utils.utils import save_state, load_state
from datasets import ClassificationDataset
from augments.cutmix import CutMix
from augments.mixup import MixUp

class ClassifcationTrainer:
    def __init__(self, model: nn.Module, optim: torch.optim, loss_fn: nn.Module, num_classes:int, device:str='cpu'):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.cutmix = CutMix(alpha=1, cutmix_rate=1.0)
        self.mixup = MixUp(alpha=8, mixup_rate=0.8)

        self.conf = ConfusionMatrix(task='multiclass',num_classes=num_classes).to(device)
        self.metrics = [
            MulticlassAccuracy(num_classes=num_classes, average='micro'),
            MulticlassF1Score(num_classes=num_classes, average='macro'),
            MulticlassAUROC(num_classes=num_classes),
        ]
        for metric in self.metrics:
            metric.to(device)
        self.scaler = torch.amp.GradScaler(device=device)


    def fit(self, num_epochs, train_db: ClassificationDataset, val_db: ClassificationDataset, batch_size, num_warmup=10, early_stop_patience=30, checkpoint_path=f'checkpoints/model.pt'):
        writer = SummaryWriter(log_dir=f"tensorboard/{self.model.__class__.__name__}/experiment")
        train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False, num_workers=2)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=0.01, end_factor=1.0, total_iters=num_warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=num_epochs - num_warmup, eta_min=0)
        # Combine both schedulers sequentially
        scheduler = torch.optim.lr_scheduler.SequentialLR(self.optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=num_epochs, eta_min=0)
        train_losses = []
        val_losses = [np.inf]

        es_counter = 0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss = self.train_epoch(train_loader)
            writer.add_scalar("Loss/train", train_loss, epoch)
            val_loss = self.evaluate(val_loader, use_progress_bar=False)
            writer.add_scalar("Loss/val", val_loss, epoch)
            scheduler.step()
            print(f'Learning Rate: {scheduler.get_last_lr()[-1]:0.4e}')
            writer.add_scalar("LearningRate", scheduler.get_last_lr()[-1], epoch)
            
            es_counter += 1
            if val_loss < min(val_losses):
                print(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                save_state(
                    checkpoint_path, 
                    self.model, 
                    self.optim,
                    info={
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'improvement_after_epochs': es_counter
                })
                es_counter = 0
            if es_counter >= early_stop_patience:
                print('Early Stopping')
                break

            train_losses.append(train_loss)
            val_losses.append(val_loss)
        self.model, self.optim = load_state(checkpoint_path, self.model, self.optim)



    def train_epoch(self, loader: DataLoader):
        self.model.train()
        losses = []

        for images, labels in tqdm(loader):
            labels = F.one_hot(labels, num_classes=self.num_classes)
            images, labels = self.cutmix.generate(images, labels)
            images, labels = self.mixup.generate(images, labels)

            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optim.zero_grad()
            
            with torch.autocast(self.device, dtype=torch.float16):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
            losses.append(loss.item())
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()

        print(f'Train Loss: {np.mean(losses):0.4f}')
        return np.mean(losses)



    def evaluate(self, loader: DataLoader, conf=False, use_progress_bar=True):
        self.model.eval()
        losses = []

        self.conf.reset()
        for metric in self.metrics:
            metric.reset() 

        data_iter = loader
        if use_progress_bar:
            data_iter = tqdm(loader, leave=False)

        with torch.inference_mode():
            for images, labels in data_iter:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                losses.append(loss.item())

                for metric in self.metrics:
                    metric.update(outputs, labels)
                self.conf.update(outputs, labels)
        print(f'Eval Loss: {np.mean(losses):0.4f}', end=' | ')
        for metric in self.metrics:
            print(f'{metric.__class__.__name__[10:]}: {metric.compute().item():0.4f}', end=' | ')
        print()
        if conf:
            print(self.conf.compute().cpu().numpy() / 1206)
        return np.mean(losses)



