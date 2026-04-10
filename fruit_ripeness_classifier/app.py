import streamlit as st
import torch
import numpy as np
import matplotlib.cm as cm
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import FruitRipenessClassifier
from src.gradcam import GradCAM

CLASS_NAMES = ['ripe', 'rotten', 'unripe']
CLASS_EMOJI = {'ripe': '✅', 'rotten': '❌', 'unripe': '🟡'}
CLASS_DESC  = {
    'ripe':   'This fruit is at peak ripeness and ready to eat.',
    'rotten': 'This fruit has gone bad and should not be consumed.',
    'unripe': 'This fruit is not yet ready — give it a few more days.'
}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FruitRipenessClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model, device

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def generate_gradcam(model, input_tensor, device):
    target_layer = model.block3[0]
    gradcam      = GradCAM(model, target_layer)
    cam, class_idx = gradcam.generate(input_tensor.to(device))
    return cam, class_idx

def overlay_heatmap(original_img, cam):
    heatmap       = cm.jet(cam)[:, :, :3]
    heatmap       = np.uint8(255 * heatmap)
    heatmap_img   = Image.fromarray(heatmap).resize((224, 224))
    original_arr  = np.array(original_img.resize((224, 224)), dtype=np.float32)
    heatmap_arr   = np.array(heatmap_img, dtype=np.float32)
    overlay       = np.uint8(np.clip(0.5 * original_arr + 0.5 * heatmap_arr, 0, 255))
    return Image.fromarray(overlay)

# --- UI ---
st.set_page_config(page_title="Fruit Ripeness Classifier", page_icon="🍎", layout="centered")

st.title("🍎 Fruit Ripeness Classifier")
st.markdown("Upload a photo of an **apple, banana, or orange** and the model will predict its ripeness stage.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    model, device = load_model()

    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs  = torch.softmax(output, dim=1)[0]

    cam, class_idx    = generate_gradcam(model, input_tensor, device)
    predicted_class   = CLASS_NAMES[class_idx]
    overlay_img       = overlay_heatmap(image, cam)

    # Results
    emoji = CLASS_EMOJI[predicted_class]
    st.markdown(f"## {emoji} Prediction: **{predicted_class.upper()}**")
    st.markdown(f"_{CLASS_DESC[predicted_class]}_")
    st.markdown("---")

    # Confidence scores
    st.markdown("### Confidence scores")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{cls}: {probs[i]*100:.1f}%")

    st.markdown("---")

    # Images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image.resize((224, 224)), caption="Original", use_container_width=True)
    with col2:
        heatmap_arr = np.uint8(255 * cm.jet(cam)[:, :, :3])
        st.image(Image.fromarray(heatmap_arr), caption="Grad-CAM Heatmap", use_container_width=True)
    with col3:
        st.image(overlay_img, caption="Overlay", use_container_width=True)

    st.markdown("---")
    st.caption("Built with PyTorch + Streamlit | CNN trained from scratch | 92% test accuracy")