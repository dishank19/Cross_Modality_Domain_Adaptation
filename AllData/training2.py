from tqdm import tqdm
import os
import time
from datetime import datetime
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils.Meter import Meter
from utils.GeneralDataset import SelfTrainingDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from itertools import cycle

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, net: nn.Module, net_name: str, criterion: nn.Module, lr: float, num_epochs: int, load_prev_model: bool = True, optimizer: torch.optim = Adam, train_dataloader = None, val_dataloader = None, self_training = False, self_training_epoch = 75, self_training_selection_portion = 0.6, unlabeled_training_dataloader = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net
        self.net = self.net.to(self.device)
        self.net_name = net_name
        self.load_prev_model = load_prev_model
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size = 25, gamma = 0.5, verbose = True)
        self.phases = ["train"] if val_dataloader is None else ["train", "val"]
        self.num_epochs = num_epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.parameter_count = count_parameters(self.net)
        self.prev_epoch = 0
        self.self_training = self_training
        self.self_training_epoch = self_training_epoch
        self.unlabeled_training_dataloader = unlabeled_training_dataloader
        self.self_training_selection_portion = self_training_selection_portion
        self.self_training_dataloader = None
        
        checkpoint_directory = os.path.join("saved_models", self.net_name)
        os.makedirs(checkpoint_directory, exist_ok = True)
        
        plots_directory = os.path.join("plots", self.net_name)
        os.makedirs(plots_directory, exist_ok = True)
        
        if self.load_prev_model:
            self.prev_epoch = self.load_model()
         
    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits
    
    def _compute_loss_and_outputs_with_selection(self, images: torch.Tensor, targets: torch.Tensor, selections: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        selections = selections.to(self.device)
        logits = self.net(images)
        confidence_scores = 2 * (torch.abs(selections - 0.5))
        loss = self.criterion(logits, targets)
        return loss
    
    def get_confidence(self, prob):
        confidence_scores = 2 * (torch.abs(prob - 0.5))
        return torch.sum(confidence_scores)
        

    def _do_epoch(self, epoch: int, phase: str):
        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        print(f"{phase} epoch: {epoch}")
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'].float(), data_batch['mask'].float()
            loss, logits = self._compute_loss_and_outputs(images, targets)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(),
                         targets.detach().cpu()
                        )
            
        epoch_loss = running_loss / total_batches
        epoch_dice, epoch_iou = meter.get_metrics_by_class()
        
        print(f"Epoch Loss: {epoch_loss}")
        print(f"Epoch Dice: {[f'Class {i + 1}: {epoch_dice[i]}' for i in range(len(epoch_dice))]}")
        print(f"Epoch IoU: {[f'Class {i + 1}: {epoch_iou[i]}' for i in range(len(epoch_iou))]}")
        print()
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.iou_scores[phase].append(epoch_iou)

        return epoch_loss
    
    def _do_epoch_with_self_training(self, epoch: int):
        self.net.train()
        meter = Meter()
        total_batches = len(self.dataloaders["train"])
        running_loss = 0.0
        running_self_loss = 0.0
        print(f"Train epoch: {epoch}")
        if len(self.self_training_dataloader) < len(self.dataloaders["train"]):
            self_training_dataloader = cycle(self.self_training_dataloader)
            labeled_training_dataloader = self.dataloaders["train"]
            length = len(self.dataloaders["train"])
        else:
            labeled_training_dataloader = cycle(self.dataloaders["train"])
            self_training_dataloader = self.self_training_dataloader
            length = len(self.self_training_dataloader)
            
        for data_batch, data_batch_self_training in zip(labeled_training_dataloader, self_training_dataloader):
            images, targets = data_batch['image'].float(), data_batch['mask'].float()
            loss, logits = self._compute_loss_and_outputs(images, targets)
            
            images_self, targets_self, selection = data_batch_self_training['image'], data_batch_self_training['mask'].float(), data_batch_self_training['selection']
            self_loss = self._compute_loss_and_outputs_with_selection(images_self, targets_self, selection)
            
            total_loss = loss + self_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())
            
            running_self_loss += self_loss.item()
            
        epoch_loss = running_loss / length
        epoch_dice, epoch_iou = meter.get_metrics_by_class()
        
        print(f"Epoch Loss: {epoch_loss}")
        print(f"Epoch Dice: {[f'Class {i + 1}: {epoch_dice[i]}' for i in range(len(epoch_dice))]}")
        print(f"Epoch IoU: {[f'Class {i + 1}: {epoch_iou[i]}' for i in range(len(epoch_iou))]}")
        print()
        
        self.losses['train'].append(epoch_loss)
        self.dice_scores['train'].append(epoch_dice)
        self.iou_scores['train'].append(epoch_iou)
        
        epoch_self_loss = running_self_loss / length
        print(f"Epoch Self Loss: {epoch_self_loss}\n\n")

        return epoch_loss
    
    def update_self_training_dataloader(self):
        print("\n\nGenerating pseudo labels for self training...")
        with torch.no_grad():
            images = []
            probs = []
            masks = []
            for data in self.unlabeled_training_dataloader:
                image = data['image'].float()
                prob = torch.sigmoid(self.net(image.to(self.device)))
                mask = (prob > 0.5).int()
                for single_image, single_prob, single_mask in zip(image, prob, mask):
                    if not torch.all(single_mask == 0):
                        images.append(single_image.detach().cpu())
                        probs.append(single_prob.detach().cpu())
                        masks.append(single_mask.detach().cpu())
                        
            print(f"Total number of unlabeled images: {len(self.unlabeled_training_dataloader) * 128}")
            print(f"Total number of images with pseudo labels: {len(images)}\n\n")
            if len(images)//5 > 0:
                confidences = [self.get_confidence(prob) for prob in probs]
                threshold = sorted(confidences, reverse = True)[len(images)//5 - 1]
                selected_images = [images[i] for i in range(len(confidences)) if confidences[i] > threshold]
                selected_masks = [masks[i] for i in range(len(confidences)) if confidences[i] > threshold]
                selected_probs = [probs[i] for i in range(len(confidences)) if confidences[i] > threshold]
                print(f"Total number of selected images : {len(selected_images)}\n\n")
                self_training_dataset = SelfTrainingDataset(selected_images, selected_masks, selected_probs)
                self.self_training_dataloader = DataLoader(self_training_dataset, batch_size = self.dataloaders['train'].batch_size, shuffle = True)
            else:
                self.self_training_dataloader = None
                
    def run(self):
        start = datetime.now()
        for epoch in range(self.prev_epoch + 1, self.num_epochs + 1):
            if self.self_training_dataloader:
                self._do_epoch_with_self_training(epoch)
            else:
                self._do_epoch(epoch, "train")
            if "val" in self.phases:
                with torch.no_grad():
                    val_loss = self._do_epoch(epoch, "val")
                
            self.scheduler.step()
            if self.self_training and epoch >= self.self_training_epoch - 1:
                self.update_self_training_dataloader()
                
            if epoch % 5 == 0:
                checkpoint = {
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
                }
                checkpoint_filename = f"{self.net_name}_epoch_{epoch}.pth"
                torch.save(checkpoint, os.path.join("saved_models", self.net_name, checkpoint_filename))
                
            print()
            self._plot_train_history()
            
            
    def _plot_graph(self, data, y_label):
        plt.figure()
        x_range = np.arange(self.prev_epoch + 1, self.prev_epoch + len(data['train']) + 1)
        plt.plot(x_range, data['train'], label = 'train', color = 'black')
        if data.get('val'):
            plt.plot(x_range, data['val'], label = 'test', color = 'green')
        plt.xlabel("epochs")
        plt.ylabel(y_label)
        plt.title(y_label)
        plt.legend()
        plt.savefig(os.path.join("plots", self.net_name, f"{y_label.replace(' ', '-')}.png"))
        plt.close()
            
    def _plot_train_history(self):
        self._plot_graph(self.losses, 'loss')
        for class_ in range(len(self.dice_scores['train'][0])):
            dice = {'train': [dice_score[class_] for dice_score in self.dice_scores['train']]}
            iou = {'train': [iou_score[class_] for iou_score in self.iou_scores['train']]}
            if self.dice_scores.get('val'):
                dice['val'] = [dice_score[class_] for dice_score in self.dice_scores['val']]
                iou['val'] = [iou_score[class_] for iou_score in self.iou_scores['val']]
            self._plot_graph(dice, f'class {class_ + 1} dice')
            self._plot_graph(iou, f'class {class_ + 1} iou')
        
    def load_model(self):
        model_name_pattern = fr"{self.net_name}_epoch_(\d+).pth"
        for file in sorted(os.listdir(os.path.join("saved_models", self.net_name)), key = lambda x: os.path.getctime(os.path.join("saved_models", self.net_name, x)), reverse = True):
            match = re.search(model_name_pattern, file)
            if match:
                previous_epoch = int(match.group(1))
                print(f"Loading {previous_epoch} epoch model...")
                checkpoint = torch.load(os.path.join("saved_models", self.net_name, file))
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.net = self.net.to(self.device)
                return previous_epoch
        print("Previous model not found. Training from begining...")
        return 0