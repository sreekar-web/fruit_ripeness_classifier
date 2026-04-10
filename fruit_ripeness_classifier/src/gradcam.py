import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FruitRipenessClassifier

CLASS_NAMES = ['ripe', 'rotten', 'unripe']

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        gradients = torch.relu(gradients)
        weights = gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        # Robust normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Fallback — use raw activations mean if gradients vanish
            cam = activations.mean(dim=1).squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), img.resize((224, 224))


def visualize_gradcam(image_path, model_path="models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FruitRipenessClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Hook into the last conv layer (block3's Conv2d)
    target_layer = model.block3[0]
    gradcam = GradCAM(model, target_layer)

    input_tensor, original_img = load_image(image_path)
    input_tensor = input_tensor.to(device)

    cam, class_idx = gradcam.generate(input_tensor)
    predicted_class = CLASS_NAMES[class_idx]

    # Overlay heatmap on image
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224, 224))

    original_array = np.array(original_img, dtype=np.float32)
    heatmap_array  = np.array(heatmap, dtype=np.float32)
    overlay = 0.5 * original_array + 0.5 * heatmap_array
    overlay = np.uint8(np.clip(overlay, 0, 255))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(original_img);      axes[0].set_title("Original");          axes[0].axis('off')
    axes[1].imshow(heatmap);           axes[1].set_title("Grad-CAM Heatmap");   axes[1].axis('off')
    axes[2].imshow(overlay);           axes[2].set_title(f"Overlay — Predicted: {predicted_class}"); axes[2].axis('off')

    plt.suptitle(f'Grad-CAM Visualization — {predicted_class.upper()}', fontsize=14)
    plt.tight_layout()

    out_path = f"outputs/gradcam_{os.path.basename(image_path)}"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {out_path}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    # Run on a few sample images from the test set
    import random
    test_dirs = {
        "ripe":   "data/test/ripe",
        "rotten": "data/test/rotten",
        "unripe": "data/test/unripe"
    }

    for class_name, folder in test_dirs.items():
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sample = random.choice(images)
        print(f"\nRunning Grad-CAM on {class_name} sample: {sample}")
        visualize_gradcam(os.path.join(folder, sample))