import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDeepfakeCNN(nn.Module):
    """
    A lightweight CNN for deepfake detection.
    Input: 128x128 RGB face image
    Output: probability of being FAKE
    """

    def __init__(self):
        super(SimpleDeepfakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))   # 32 -> 16

        x = x.view(-1, 128 * 16 * 16)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.fc2(x))  # probability of FAKE
        return x

# ---------- Inference Wrapper Function ---------- #

def load_cnn_model(model_path=None, device=None):
    """
    Loads a trained CNN model.
    If no model_path is provided, returns a randomly initialized model.
    """

    model = SimpleDeepfakeCNN()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    return model


def predict_frame(model, face):
    """
    Runs deepfake prediction on a single face frame.

    Args:
        model: loaded CNN model
        face: numpy array (128x128x3, normalized)

    Returns:
        float: fake probability (0–1)
    """

    device = next(model.parameters()).device

    # Convert face to tensor (B x C x H x W)
    face_tensor = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        prob_fake = model(face_tensor).item()

    return float(prob_fake)
