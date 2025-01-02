import torch
from model1 import PneumoniaModel
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PneumoniaModel().to(device)
    model.load_state_dict(torch.load('pneumonia_resnet18.pth'))
    model.eval()

    _, _, test_loader = get_dataloaders(data_dir)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Calculate metrics
    accuracy = np.trace(cm) / np.sum(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    data_dir = 'chest_xray/'
    evaluate_model(data_dir)