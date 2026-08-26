"""
Gradio demo for PlantVillage crop-disease classification.
Loads the final fine-tuned ResNet-50 (100% fraction) and shows:
predicted class, calibrated confidence, top-3 alternatives, and a Grad-CAM overlay.

Run with:  python app.py
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "new_saved_models/resnet50_unfrozen_best_state_100p.pth"

# Temperature fitted in Deliverable 3 (confidence calibration) — update if the model is retrained
TEMPERATURE = 0.615

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Verified class order (matches sorted folder names / class_to_idx built during training)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = len(CLASS_NAMES)


# ------------------------------------------------------------------
# Model (same architecture as build_model() in the training notebook)
# ------------------------------------------------------------------
def build_model(n_classes, dropout=0.3):
    model = models.resnet50(weights=None)  # weights are overwritten by our checkpoint
    input_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(input_features, n_classes))
    return model


model = build_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

gradcam_target_layer = model.layer4[-1]


# ------------------------------------------------------------------
# Grad-CAM (same hook logic as the notebook)
# ------------------------------------------------------------------
def grad_cam(x, cls, layer):
    activations = {}
    x = x.detach().requires_grad_(True)

    def forward_hook(module, inputs, output):
        output.retain_grad()
        activations["value"] = output

    hook = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad()
        output = model(x)
        output[0, cls].backward()

        activation = activations["value"][0].detach()
        gradient = activations["value"].grad[0].detach()

        weights = gradient.mean((1, 2))
        cam = torch.relu((weights[:, None, None] * activation).sum(0))
        cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    finally:
        hook.remove()

    return cam


def overlay_cam_on_image(pil_image, cam):
    import matplotlib.cm as cm

    cam_up = F.interpolate(
        torch.tensor(cam)[None, None], size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    base = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0

    cam_normalized = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
    heatmap = cm.jet(cam_normalized)[..., :3]

    overlay = 0.5 * base + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ------------------------------------------------------------------
# Prediction function wired into the Gradio interface
# ------------------------------------------------------------------
def predict(pil_image):
    if pil_image is None:
        return None, None

    pil_image = pil_image.convert("RGB")
    input_tensor = eval_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)

    calibrated_probs = F.softmax(logits / TEMPERATURE, dim=1)[0].cpu().numpy()

    top3_idx = calibrated_probs.argsort()[::-1][:3]
    top3 = {CLASS_NAMES[i]: float(calibrated_probs[i]) for i in top3_idx}

    pred_class = int(top3_idx[0])
    cam = grad_cam(input_tensor, pred_class, gradcam_target_layer)
    cam_overlay = overlay_cam_on_image(pil_image, cam)

    return top3, cam_overlay


# ------------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a leaf photo"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top-3 predictions (calibrated confidence)"),
        gr.Image(type="numpy", label="Grad-CAM — where the model is looking"),
    ],
    title="Crop Disease Classifier (ResNet-50, fine-tuned)",
    description=(
        "Upload a leaf image to get a disease prediction with calibrated confidence "
        "and a Grad-CAM overlay showing which region drove the prediction. "
        "This is a decision-support prototype, not a treatment-prescription system."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
