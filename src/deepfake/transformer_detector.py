import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ---------------------------------------------------------
# Helper: Patch Embedding (turns image into patch tokens)
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, emb_dim=256):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # → (B, emb_dim, H/ps, W/ps)
        x = x.flatten(2)        # → (B, emb_dim, num_patches)
        x = x.transpose(1, 2)   # → (B, num_patches, emb_dim)
        return x


# ---------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# ---------------------------------------------------------
# Vision Transformer for Deepfake Detection
# ---------------------------------------------------------
class DeepfakeTransformer(nn.Module):
    """
    Processes a *sequence* of face frames.
    Input shape: (B, T, 3, 128, 128)
    Output: probability video is FAKE.
    """

    def __init__(self, img_size=128, patch_size=16, emb_dim=256,
                 depth=4, num_heads=4, ff_dim=512, num_frames=16):

        super().__init__()

        self.num_frames = num_frames

        # Patch embedding for each frame
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_dim)

        # Positional encoding for patches
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))

        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, ff_dim) for _ in range(depth)
        ])

        # Temporal classifier
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, video_frames):
        """
        Args:
            video_frames: (B, T, C, H, W)
        """

        B, T, C, H, W = video_frames.shape
        frame_logits = []

        for t in range(T):
            frame = video_frames[:, t]  # (B, C, H, W)

            # Patch embedding → tokens
            tokens = self.patch_embed(frame)  # (B, num_patches, emb_dim)

            # Add positional embeddings
            tokens = tokens + self.pos_embed

            # Pass through transformer layers
            for block in self.transformer_blocks:
                tokens = block(tokens)

            # Global average pooling over patches
            frame_repr = tokens.mean(dim=1)  # (B, emb_dim)

            # Predict FAKE probability for this frame
            frame_logit = torch.sigmoid(self.fc(frame_repr))  # (B, 1)
            frame_logits.append(frame_logit)

        # Stack frame-level predictions → (B, T, 1)
        frame_logits = torch.stack(frame_logits, dim=1)

        # Final video-level probability: average across frames
        video_pred = frame_logits.mean(dim=1)  # (B, 1)

        return video_pred


# ---------------------------------------------------------
# Inference Wrapper Function
# ---------------------------------------------------------
def load_transformer_model(model_path=None, device=None):
    model = DeepfakeTransformer()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    return model


def predict_video_sequence(model, frames):
    """
    Args:
        model: transformer model
        frames: list of face frames normalized (0–1), shape → T x 128 x 128 x 3

    Returns:
        float: probability video is FAKE
    """

    device = next(model.parameters()).device

    # Pick up to 16 frames evenly spaced
    T = min(len(frames), 16)
    idxs = np.linspace(0, len(frames)-1, T, dtype=int)

    selected_frames = [frames[i] for i in idxs]

    # Convert to tensor format (B=1)
    tensor_frames = torch.tensor(selected_frames, dtype=torch.float32)
    tensor_frames = tensor_frames.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)

    tensor_frames = tensor_frames.to(device)

    with torch.no_grad():
        pred = model(tensor_frames).item()

    return float(pred)
