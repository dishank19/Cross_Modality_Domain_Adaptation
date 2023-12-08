import os
from models.UNet2d import UNet2d
from torch.optim import Adam
from utils.Meter import BCEDiceLoss
from training import Trainer
from data_processing import get_datasets
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from utils.GeneralDataset import UnlabeledDataset

import warnings
warnings.simplefilter("ignore")

def is_blank(mask_path):
    mask = np.array(Image.open(mask_path))
    return np.all(mask == 0)
    
def get_training_images_and_masks():
    training_path = os.path.join('BRATS_DATA', '2D', 'Training')
    images = [(os.path.join(training_path, image_dir, image), ) for image_dir in os.listdir(training_path) for image in os.listdir(os.path.join(training_path, image_dir)) if image_dir.endswith("t1n'")]
    
    images = [image for image in images if not is_blank(image[0].replace("-t1n'", '-seg'))]
    
    masks = [image[0].replace("-t1n'", '-seg') for image in images]
    
    return images, masks

def get_testing_images_and_masks():
    training_path = os.path.join('BRATS_DATA', '2D', 'Testing')
    images = [(os.path.join(training_path, image_dir, image), ) for image_dir in os.listdir(training_path) for image in os.listdir(os.path.join(training_path, image_dir)) if image_dir.endswith('t2f')]
    
    images = [image for image in images if not is_blank(image[0].replace('-t2f', '-seg'))]
    
    masks = [image[0].replace('-t2f', '-seg') for image in images]
    
    return images, masks

def get_unlabeled_training_images():
    training_path = os.path.join('BRATS_DATA', '2D', 'Training')
    images = [(os.path.join(training_path, image_dir, image), ) for image_dir in os.listdir(training_path) for image in os.listdir(os.path.join(training_path, image_dir)) if image_dir.endswith('t2f')]
    
    return images

unlabeled_training_dataloader = DataLoader(UnlabeledDataset(images = get_unlabeled_training_images(), center_crop_size = (224, 224)), batch_size = 128, pin_memory = True, shuffle = False)

train_dataloader, val_dataloader = get_datasets(get_training_images_and_masks, get_testing_images_and_masks, batch_size = 64, center_crop_size = (224, 224))

model = {"model": UNet2d(in_channels = 1, n_classes = 1, n_channels = 16).to('cuda'), "name":"trainingTranslation"}

trainer = Trainer(net = model["model"],
                  net_name = model["name"],
                  criterion = BCEDiceLoss(),
                  lr = 1e-3,
                  num_epochs = 180,
                  optimizer = Adam,
                  load_prev_model = True,
                  train_dataloader = train_dataloader,
                  val_dataloader = val_dataloader,
                  self_training = True,
                  self_training_epoch = 150,
                  self_training_selection_portion = 0.6,
                  unlabeled_training_dataloader = unlabeled_training_dataloader)
    
trainer.run()