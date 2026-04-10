# 🍎 Fruit Ripeness Classifier

A convolutional neural network built **from scratch** in PyTorch that classifies 
fruits as **ripe, unripe, or rotten** from a single image.

## Results
| Metric | Score |
|--------|-------|
| Val Accuracy | 94.05% |
| Test Accuracy | 92.35% |
| Macro F1 | 0.92 |

## Demo
![App Screenshot](outputs/app_screenshot.png)

## Grad-CAM Visualizations
The model uses Grad-CAM to highlight which regions of the image influenced 
the prediction.

![Grad-CAM](outputs/gradcam_ripe.png)
![Grad-CAM](outputs/gradcam_rotten.png)
![Grad-CAM](outputs/gradcam_unripe.png)

## Model Architecture
- 3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool → Dropout)
- Fully connected classifier head
- ~13,000 training images across 3 classes
- Trained with class-weighted CrossEntropyLoss to handle imbalance

## Tech Stack
- PyTorch, torchvision
- scikit-learn, seaborn (evaluation)
- Streamlit (demo app)

## Project Structure
```
fruit_ripeness_classifier/
├── src/
│   ├── model.py       # CNN architecture
│   ├── dataset.py     # Data loading and augmentation
│   ├── train.py       # Training loop
│   ├── evaluate.py    # Metrics and confusion matrix
│   └── gradcam.py     # Grad-CAM implementation
├── app.py             # Streamlit demo
└── outputs/           # Confusion matrix and Grad-CAM plots
```

## Run Locally
```bash
git clone https://github.com/yourusername/fruit-ripeness-classifier
cd fruit_ripeness_classifier
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten) — 39.9k images across 9 fruit/ripeness combinations, remapped to 3 ripeness classes.