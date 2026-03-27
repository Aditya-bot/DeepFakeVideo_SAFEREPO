from xml.parsers.expat import model

import torch
import torch.nn as nn
from torchvision import models


# ================= MODEL =================
class ResNetDeepfake(nn.Module):
    def __init__(self):
        super(ResNetDeepfake, self).__init__()

        self.model = models.resnet18(pretrained=True)

        num_features = self.model.fc.in_features

        # REMOVE original classifier
        self.model.fc = nn.Identity()

        # New classifier
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x, return_features=False):

        features = self.model(x)  # (batch, 512)

        if return_features:
            return features

        out = self.classifier(features)
        return out


# ================= LOAD MODEL =================
def load_cnn_model(model_path=None, device=None):

    model = ResNetDeepfake()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)

        model_dict = model.state_dict()

        # Filter matching keys only
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        model_dict.update(filtered_dict)

        model.load_state_dict(model_dict)
        model.eval()

    return model


# ================= INFERENCE =================
def predict_frame(model, face):
    """
    face: numpy array (H, W, 3)
    """

    device = next(model.parameters()).device

    # Convert to tensor
    face_tensor = torch.tensor(face, dtype=torch.float32)\
        .permute(2, 0, 1)\
        .unsqueeze(0)

    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        output = model(face_tensor)
        prob_fake = torch.sigmoid(output).item()  # IMPORTANT

    return float(prob_fake)