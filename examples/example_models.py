import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


def show_image(image, text=None, activation=None):
    if text is not None:
        plt.suptitle(text, fontsize=12)
    if activation is not None:
        plt.title('Activation: {:.2f}'.format(activation), fontsize=8)

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def show_image_grid(images, texts=None, activations=None, grid_size=(4, 4), image_size=(4, 4), spacing=0.1):
    num_images = len(images)
    num_rows, num_cols = grid_size

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * image_size[0], num_rows * image_size[1]))

    for i, (image, text, activation) in enumerate(zip(images, texts, activations)):
        ax = axes[i // num_cols, i % num_cols]
        if text is not None:
            ax.set_title(text, fontsize=12)
        if activation is not None:
            ax.set_xlabel('Activation: {:.2f}'.format(activation), fontsize=8)
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()


class _ExampleModel(nn.Module):
    def __init__(self, name='model', device='cpu', criterion=None, optimizer=None):
        super(_ExampleModel, self).__init__()
        self.name = name
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device
        self.to(device)

    def train(self, epochs=10):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device).type(type(next(self.parameters()).data))
                labels = labels.to(self.device).type(torch.LongTensor)

                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {running_loss/100:.4f} on {self.name}")
                    running_loss = 0.0

    def evaluate(self):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device).type(type(next(self.parameters()).data))
                labels = labels.to(self.device).type(torch.LongTensor)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy on {self.name}: {accuracy:.2f}%")

    def save(self):
        torch.save(self.state_dict(), f"./data/{self.name}.pth")
        print(f"Saved {self.name}.")

    def load(self):
        import os
        if os.path.isfile(f"./data/{self.name}.pth"):
            print(f"Loading {self.name}...")
            self.load_state_dict(torch.load(f"./data/{self.name}.pth"))


class MNIST_model(_ExampleModel):
    def __init__(self, name='model', device='cpu', load=False):
        super(MNIST_model, self).__init__(name=name, device=device)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        transformer = Compose([
            ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        # Load MNIST dataset
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transformer)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)

        # Create data loaders
        batch_size = 128
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        if load:
            self.load()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CIFAR_model(_ExampleModel):
    def __init__(self, name='model', device='cuda', load=False, KERNEL_SIZE=(3, 3)):
        super(CIFAR_model, self).__init__(name=name, device=device)

        self.conv1 = nn.Conv2d(3, 32, KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, KERNEL_SIZE, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(64, 128, KERNEL_SIZE, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, KERNEL_SIZE, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)

        self.groupnorm = nn.GroupNorm(32, 128)
        self.activation = nn.SiLU()
        self.zeropad1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(128, 16, 3, stride=1)
        self.zeropad2 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(16, 8, 1, stride=1)

        self.flatten = nn.Flatten()
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(8 * 6 * 6, 128)
        self.dropout4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        transformer = Compose([
            ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        # Load MNIST dataset
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transformer)

        # Create data loaders
        batch_size = 128
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if load:
            self.load()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = self.groupnorm(x)
        x = self.activation(x)
        x = self.zeropad1(x)
        x = self.conv7(x)
        x = self.zeropad2(x)
        x = self.conv8(x)

        x = self.flatten(x)
        x = self.dropout3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)


class MNIST_autoencoder(_ExampleModel):
    def __init__(self, name='model', device='cuda', load=False):
        super(MNIST_autoencoder, self).__init__(name=name, device=device)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        transformer = Compose([
            ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        train_dataset = MNIST(root='./data', train=True, download=True, transform=transformer)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)
        batch_size = 128
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if load:
            self.load()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def train(self, epochs=10):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device).type(type(next(self.parameters()).data))

                self.optimizer.zero_grad()

                output = self(data)
                loss = self.criterion(output, data)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if i % 100 == 99:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {running_loss/100:.4f} on {self.name}")
                    running_loss = 0.0

    def evaluate(self):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device).type(type(next(self.parameters()).data))

                output = self(data)
                loss = self.criterion(output, data)
                total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)
        print(f"Average Reconstruction Loss on {self.name}: {average_loss:.4f}")




