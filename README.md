# SceneShield: AI Fake Image Classifier

![SceneShield AI Fake Image Classifier](images/banner.png)

## Overview
This repository contains a deep learning model for **fake scene classification**, utilizing data from **CIDAUT AI Fake Scene Classification 2024**. The notebook includes Python code for loading data, applying data augmentations, and training a model using **PyTorch**.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Loading](#data-loading)
4. [Data Augmentations](#data-augmentations)
5. [Advanced Custom Dataset Class](#advanced-custom-dataset-class)
6. [Running the Code](#running-the-code)
7. [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
8. [Results](#results)
9. [Usage](#usage)
10. [License](#license)

## Introduction
Fake scene classification is a challenging task in computer vision. This project aims to detect fake scenes using advanced **deep learning techniques** and **data augmentation** strategies. 

## Installation
To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Loading
The dataset is loaded and preprocessed using **PyTorch DataLoader**.

```python
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Data Augmentations
We apply various augmentations to improve model generalization:

- **Random Resizing**
- **Horizontal Flipping**
- **Color Jittering**
- **Gaussian Noise Injection**

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor()
])
```

![Data Augmentation Example](images/augmentation.png)

## Advanced Custom Dataset Class
A custom dataset class is implemented for more control over data processing:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
```

## Running the Code
To train the model, execute:

```bash
python train.py
```

## Gaussian Mixture Model (GMM)
A **GMM-based approach** is implemented to analyze clusters of fake and real images:

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(features)
```

![GMM Clusters](images/gmm.png)

## Results
The final trained model achieves **97% accuracy** on the validation set.

## Usage
To test a new image, run:

```bash
python predict.py --image sample.jpg
```

## License
This project is licensed under the **MIT License**.

---
**Note:** This code is intended for educational and research purposes only and may not be optimized for production use.
