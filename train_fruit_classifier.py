import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os

# Parameters
batch_size = 32
num_epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Device used: {device}")

# Load transformations
weights = EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

# Load dataset
dataset_path = 'Fruit_Dataset'

if not os.path.exists(dataset_path):
    print(f"❌ Directory '{dataset_path}' does not exist!")
    exit()

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define class names
CLASS_NAMES = dataset.classes

# Diagnostic display
print(f"📊 Number of classes: {len(CLASS_NAMES)}")
print(f"📸 Total images: {len(dataset)}")
print("📂 Classes:", CLASS_NAMES)

if len(dataset) == 0:
    print("❌ No images found. Check the contents of 'Fruit_Dataset'.")
    exit()

# Load EfficientNet model
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("🚀 Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataset)
    epoch_acc = running_corrects.double() / len(dataset)
    print(f"📈 Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

# Save model
torch.save(model.state_dict(), 'fruit_classifier_model.pt')
print("✅ Model trained and saved as fruit_classifier_model.pt")