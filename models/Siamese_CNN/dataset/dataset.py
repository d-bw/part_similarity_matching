import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import os


def convert_images_to_grayscale(root_dir):
    # Walk through the directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                # Open the image
                img = Image.open(file_path)
                # Convert the image to grayscale
                img = img.convert('L')
                # Save the image back
                img.save(file_path)
                print(f"Converted {file_path} to grayscale")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

def convert_non_bmp_to_bmp(root_dir):
    # Walk through the directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if not file.lower().endswith('.bmp'):
                try:
                    # Open the image
                    img = Image.open(file_path)
                    # Create new file name with .bmp extension
                    new_file_path = os.path.splitext(file_path)[0] + '.bmp'
                    # Save the image in .bmp format
                    img.save(new_file_path)
                    # Remove the old file
                    os.remove(file_path)
                    print(f"Converted {file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


def imshow(img,text=None,should_save=False):
    #展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy() #将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
    plt.show()

def show_plot(iteration,loss):
    #绘制损失变化图
    plt.plot(iteration,loss)
    plt.show()


class SiameseNetworkDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, should_invert=True):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.should_invert = should_invert
        self.class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.dataset_dir, class_dir)
            reference_path = os.path.join(class_path, 'reference')
            if os.path.exists(reference_path):
                reference_images = [os.path.join(reference_path, img) for img in os.listdir(reference_path) if img.endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
                other_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.bmp', '.jpg', '.jpeg', '.png')) and img != 'reference']
                data.append((class_dir, reference_images, other_images))
        return data

    def __getitem__(self, index):
        # Randomly select a class
        class_dir, reference_images, other_images = random.choice(self.data)

        # Randomly select img0 from other_images
        img0_path = random.choice(other_images)

        # Determine if we should get a same-class or different-class image
        should_get_same_class = random.randint(0, 1)  # 50% chance

        if should_get_same_class:
            img1_path = random.choice(reference_images)
            other_class_dir = class_dir  # Same class
        else:
            while True:
                other_class_dir, other_reference_images, _ = random.choice(self.data)
                if other_class_dir != class_dir:
                    img1_path = random.choice(other_reference_images)
                    break

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(other_class_dir != class_dir)], dtype=np.float32))

    def __len__(self):
        return sum(len(other_images) for _, _, other_images in self.data)

