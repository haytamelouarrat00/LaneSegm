import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import os
from roboflow import Roboflow

rf = Roboflow(api_key="J09U3OiEmcboSrbUKAgf")
project = rf.workspace("#####").project("fsai-11-03-2024") #Private Roboflow key
version = project.version(1)
dataset = version.download("coco")


class FastSCNN(nn.Module):
    def __init__(self, input_channel, num_classes, dropout_rate=0.2):
        super().__init__()

        # Enhanced modules with more flexible configurations
        self.learning_to_downsample = Learning2Downsample(input_channel)
        self.global_feature_extractor = GlobalFeatureExtractor(channels=[64, 96, 128])
        self.feature_fusion = FeatureFusionModule(low_channels=64, high_channels=128)
        self.classifier = Classifier(num_classes, dropout_rate)

    def forward(self, x):
        shared = self.learning_to_downsample(x)
        global_features = self.global_feature_extractor(shared)
        fused_features = self.feature_fusion(shared, global_features)
        return self.classifier(fused_features)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Learning2Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        )

    def forward(self, x):
        return self.conv_layers(x)


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, channels=[64, 96, 128], expansion_ratio=6):
        super().__init__()
        self.stages = nn.ModuleList([
            self._make_stage(channels[0], channels[0], expansion_ratio),
            self._make_stage(channels[0], channels[1], expansion_ratio),
            self._make_stage(channels[1], channels[2], expansion_ratio)
        ])
        self.ppm = PyramidPoolingModule(channels[2], channels[2])

    def _make_stage(self, in_channels, out_channels, expansion_ratio):
        return nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=2, expand_ratio=expansion_ratio),
            *[InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=expansion_ratio) for _ in range(2)]
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.ppm(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_features, out_features, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_features, in_features // len(sizes), kernel_size=1, bias=False)
            ) for size in sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features * (len(sizes) + 1), out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages] + [x]
        return self.bottleneck(torch.cat(priors, 1))


class FeatureFusionModule(nn.Module):
    def __init__(self, low_channels=64, high_channels=128):
        super().__init__()
        self.low_proj = nn.Conv2d(low_channels, high_channels, kernel_size=1)
        self.high_proj = nn.Conv2d(high_channels, high_channels, kernel_size=1)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, kernel_size=3, padding=1, groups=high_channels),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_res_input, high_res_input):
        low_res_input = F.interpolate(low_res_input, scale_factor=4, mode='bilinear', align_corners=True)
        low_res_input = self.low_proj(low_res_input)
        high_res_input = self.high_proj(high_res_input)
        x = low_res_input + high_res_input
        return self.fuse_conv(x)


class Classifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classifier(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.stride = stride
        self.use_residual = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class FSOCODataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, early_stopping_patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


def main():
    # Hyperparameters
    input_channels = 3
    num_classes = 5
    learning_rate = 0.001
    batch_size = 32
    epochs = 50
    early_stopping_patience = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset preparation
    dataset_path = "path_to_dataset"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Model initialization
    model = FastSCNN(input_channel=input_channels, num_classes=num_classes).to(device)

    # Weight initialization
    def initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    model.apply(initialize_weights)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping_patience)

    # Evaluate on test set
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    else:
        print("No saved model found, evaluating with the current model")

    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()