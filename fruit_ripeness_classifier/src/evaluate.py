import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders
from src.model import FruitRipenessClassifier

def evaluate(data_dir, model_path="models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_dataloaders(data_dir, batch_size=32)
    print(f"Classes: {classes}")

    model = FruitRipenessClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix — Fruit Ripeness Classifier')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    # Overall test accuracy
    test_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    evaluate(data_dir="data")