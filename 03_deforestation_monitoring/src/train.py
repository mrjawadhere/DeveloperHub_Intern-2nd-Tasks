import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import models
from tqdm import tqdm

class NpyDataset(Dataset):
    def __init__(self, data_dir, labels_file):
        self.paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        # labels_file: CSV with filename,label (0 forest,1 deforested)
        self.labels = {row[0]: int(row[1]) for row in np.loadtxt(labels_file, delimiter=',', dtype=str)}
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        fname = os.path.basename(path)
        x = np.load(path)[:3]  # first 3 bands
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(self.labels.get(fname, 0), dtype=torch.long)
        return x, y

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features.fc = nn.Linear(self.features.fc.in_features, 2)
    def forward(self, x):
        return self.features(x)

def train(args):
    dataset = NpyDataset(args.data_dir, args.labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--labels', required=True, help='CSV file of labels')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    train(args)

if __name__ == '__main__':
    main()
