import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
import matplotlib.pyplot as plt
import os


class LeNetPathMNISTModified(nn.Module):
    def __init__(self, num_classes=9):
        super(LeNetPathMNISTModified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        feature_size = 32 * 1 * 1
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        labels = labels.squeeze().long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
        optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.squeeze().long()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './Data'))
    
    train_dataset = PathMNIST(split='train', root=root_path, download=True, transform=transform, as_rgb=True)
    test_dataset = PathMNIST(split='test', root=root_path, download=True, transform=transform, as_rgb=True)
    
    train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False)
    
    model = LeNetPathMNISTModified().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Centralized Training - Adam lr=0.001 + Gradient Clipping')
    plt.grid(True)
    plt.savefig('accuracy_plot.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()