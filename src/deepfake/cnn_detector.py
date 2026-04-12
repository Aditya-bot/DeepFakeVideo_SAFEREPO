import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


# ================= MODEL =================
class ResNetDeepfake(nn.Module):
    def __init__(self):
        super(ResNetDeepfake, self).__init__()

        # Load pretrained ResNet18 backbone
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        num_features = self.model.fc.in_features

        # Remove original classifier
        self.model.fc = nn.Identity()

        # Binary classifier head
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x, return_features=False):
        features = self.model(x)

        if return_features:
            return features

        out = self.classifier(features)   # logits
        return out


# ================= INFERENCE TRANSFORM =================
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================= LOAD MODEL =================
def load_cnn_model(
    model_path="models/cnn_deepfake.pt",
    device=None
):
    """
    Loads trained CNN model.
    Defaults to trained model path.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CNN model from: {model_path}")

    model = ResNetDeepfake().to(device)

    if model_path is not None:
        state_dict = torch.load(
            model_path,
            map_location=device
        )

        model_dict = model.state_dict()

        # Load only matching keys
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict
        }

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

    model.eval()

    return model


# ================= FRAME PREDICTION =================
def predict_frame(model, face):
    """
    face: numpy array (H, W, 3) in BGR normalized [0,1]
    returns: probability of REAL (0–1)
    """

    device = next(model.parameters()).device

    # convert normalized face back to uint8
    face = (face * 255).astype(np.uint8)

    # BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # apply deterministic inference transform
    face_tensor = inference_transform(face)\
        .unsqueeze(0)\
        .to(device)

    model.eval()

    with torch.no_grad():
        logits = model(face_tensor)

        # convert logits → fake probability

        prob_real = torch.sigmoid(logits).item()

    return float(prob_real)


# ================= VIDEO-LEVEL PREDICTION =================
def predict_video(model, face_frames, max_frames=30):
    """
    face_frames: list of face crops
    returns: averaged REAL score
    """

    if len(face_frames) == 0:
        return 0.5

    # deterministic frame selection
    idxs = np.linspace(
        0,
        len(face_frames) - 1,
        min(max_frames, len(face_frames)),
        dtype=int
    )

    selected_frames = [face_frames[i] for i in idxs]

    scores = []

    for frame in selected_frames:
        score = predict_frame(model, frame)
        scores.append(score)

    return float(np.mean(scores))