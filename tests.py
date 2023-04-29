from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms


def class_accuracy(net, test_dataset, image_resize=224):
    print('Testing class accuracy...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    net.eval()
    acc = 0
    tests = 0
    for images, labels in tqdm(test_dataloader, position=0, leave=False):
        images, labels = images.to(device), labels.to(device) # Move to device
        if image_resize is not None:
            images = transforms.Resize(image_resize)(images)
        image, label = images[0], labels[0].item()
        pred = net.predict(image)
        tests += 1
        if pred == label:
            acc += 1
    acc /= tests
    print(f'Accuracy: {acc*100:.2f}%')


def mae(net, test_dataset, image_resize=224):
    print('Testing MAE...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    net.eval()
    mae = 0
    tests = 0
    for images, labels in tqdm(test_dataloader, position=0, leave=False):
        images, labels = images.to(device), labels.to(device) # Move to device
        if image_resize is not None:
            images = transforms.Resize(image_resize)(images)
        image, label = images[0], labels[0].item()
        pred = net.predict(image)
        tests += 1
        mae += abs(label - pred)
    mae /= tests
    print(f'MAE: {mae:.4g}')


def confusion_matrix(net, test_dataset, num_classes=2, image_resize=224):
    print('Testing class accuracy...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    net.eval()
    confusion = np.zeros((num_classes, num_classes))
    for images, labels in tqdm(test_dataloader, position=0, leave=False):
        images, labels = images.to(device), labels.to(device) # Move to device
        if image_resize is not None:
            images = transforms.Resize(image_resize)(images)
        image, label = images[0], labels[0].item()
        pred = net.predict(image)
        confusion[label][pred] += 1

    print('\nConfusion matrix (col: pred, row: actual/label):')
    for i in range(num_classes):
        print(f'{i:11}', end='')
    print('')
    for i in range(num_classes):
        print(f'{i}    ', end='')
        for j in range(num_classes):
            print(f'[{confusion[i][j]:9.0f}]', end='')
        print('')

    print('\nAccuracy per class:')
    for i in range(num_classes):
        print(f'Class {i}: {confusion[i][i]/confusion[i].sum()*100:.2f}%')