import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import sys

# Define class names
CLASS_NAMES = ["Apple", "Banana", "Carrot", "Grape", "Guava", "Jujube", "Mango", "Orange"]

# Check if image path is provided
if len(sys.argv) != 2:
    print("❗ Usage: python fruit_classifier.py image_path.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Modify the final layer to match the number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
model.load_state_dict(torch.load('fruit_classifier_model.pt', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = weights.transforms()
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(img_tensor)
    prob = torch.nn.functional.softmax(output, dim=1)
    conf, pred = torch.max(prob, 1)
    predicted_class = CLASS_NAMES[pred.item()]
    confidence = conf.item()

# Display result
print(f"📷 Image: {image_path}")
print(f"✅ Prediction: {predicted_class} ({confidence*100:.2f}%)")