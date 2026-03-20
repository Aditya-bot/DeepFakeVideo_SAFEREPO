import torch
import torch.nn as nn
from torchvision import models


# ================= MODEL =================
class ResNetDeepfake(nn.Module):
    def __init__(self):
        super(ResNetDeepfake, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # binary output

    def forward(self, x):
        return self.model(x)


# ================= LOAD MODEL =================
def load_cnn_model(model_path=None, device=None):

    model = ResNetDeepfake()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
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