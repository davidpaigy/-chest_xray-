import torch
import torch.nn as nn
import torch.optim as optim
from model1 import PneumoniaModel
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

def train_model(data_dir, epochs=20, batch_size=32, learning_rate=0.001, step_size=7, gamma=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PneumoniaModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot the loss curve
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'pneumonia_resnet18.pth')

if __name__ == '__main__':
    data_dir = 'chest_xray/'
    train_model(data_dir)
