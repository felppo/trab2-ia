import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn

import torchvision
from torchvision import transforms as tf
from torchvision import datasets as dts
from torch.utils.data import DataLoader
import torchvision.models as models

import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("usando", device)

model = models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model.classifier[3] = nn.Linear(in_features=1280, out_features=10)
model.to(device)

# ----------------------------------------------------------------------------------------------------------------------

trans = tf.Compose([
    # tf.Resize(size=(224, 224)),
    tf.Resize(size=(256, 256)),
    tf.CenterCrop(224),
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = dts.ImageFolder(root="dataset/train/", transform=trans)
train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)

valid_dataset = dts.ImageFolder(root="dataset/valid/", transform=trans)
valid_loader = DataLoader(valid_dataset, batch_size=6, shuffle=True)

# ----------------------------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

num_epochs = 50
patience = 10
patience_count = 0

best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())

epoch = 0
stop = False

loss_train = []
loss_val = []

# ----------------------------------------------------------------------------------------------------------------------

while not stop:
    # Modelo em modo de treinamento
    model.train()

    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    loss_train.append(avg_train_loss)

    # Modelo em modo de validação
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(valid_loader)
    loss_val.append(avg_val_loss)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Accuracy: {accuracy:.2f}%")

# ----------------------------------------------------------------------------------------------------------------------

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_count = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"Parada após {epoch + 1} épocas.")
            break

    epoch += 1

    if epoch == num_epochs:
        stop = True

# ----------------------------------------------------------------------------------------------------------------------

x = list(range(len(loss_train)))
plt.plot(x, loss_train, marker='o', label='Loss de Treino', color='blue')
plt.plot(x, loss_val, marker='s', label='Loss de Validação', color='orange')

# Decorations
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model.load_state_dict(best_model_state)

test_dataset = dts.ImageFolder(root="dataset/test/", transform=trans)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

# Teste
model.eval()

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        f1 = f1_score(all_labels, all_preds, average="weighted")

accuracy = 100 * correct / total

print(f"\nAccuracy: {accuracy:.2f}%\nF1-Score: {f1:.2f}")

# ----------------------------------------------------------------------------------------------------------------------

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.xlabel("Predição")
plt.ylabel("Classificação")
plt.tight_layout()
plt.show()