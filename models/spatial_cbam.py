import torch
import torch.nn as nn


class SpatialAttentionSubModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att


class SpatialAttentionRefinementModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_attention = SpatialAttentionSubModule(kernel_size=kernel_size)

    def forward(self, features):
        return [self.spatial_attention(f) for f in features]


if __name__ == "__main__":
    spatial_refinement = SpatialAttentionRefinementModule()

    test_feats = [
        torch.randn(2, 256, 128, 128),
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16)
    ]

    output_feats = spatial_refinement(test_feats)

    print("Spatial Attention Refinement Module Output:")
    for i, f in enumerate(output_feats):
        print(f"Level {i + 2} Output Shape: {f.shape}")