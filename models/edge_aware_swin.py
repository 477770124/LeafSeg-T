import torch
import torch.nn as nn
import timm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import os


class EdgeAwareMechanism(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=False)
        self.c = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_spatial = x.transpose(1, 2).view(B, C, H, W)

        phi = self.sigmoid(self.edge_conv(x_spatial))

        phi_flat = phi.view(B, 1, L).transpose(1, 2)

        edge_bias = self.c * phi_flat
        return edge_bias


class ChannelAttentionSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


@BACKBONE_REGISTRY.register()
class SwinBackbone(Backbone):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        self.window_size = 8
        self.swin_model_name = "swin_tiny_patch4_window7_224"
        self.out_indices = (0, 1, 2, 3)
        self.out_channels = [96, 192, 384, 768]

        self.input_size = (500, 530)
        self.pretrained_weight_path = r"D:\1Project\leaf_detectron2\weights\swin_tiny_patch4_window7_224.pth"

        self.swin = timm.create_model(
            self.swin_model_name,
            pretrained=False,
            features_only=True,
            out_indices=self.out_indices,
            window_size=self.window_size,
            img_size=self.input_size,
            norm_layer=torch.nn.Identity
        )

        self._inject_edge_aware_mechanism()

        self._load_pretrained()

        self.se_layers = nn.ModuleList([
            ChannelAttentionSE(chan, reduction=16) for chan in self.out_channels
        ])

    def _inject_edge_aware_mechanism(self):
        self.edge_extractors = nn.ModuleList([
            EdgeAwareMechanism(dim) for dim in self.out_channels
        ])

    def _load_pretrained(self):
        if os.path.exists(self.pretrained_weight_path):
            print(f"正在加载本地预训练权重: {self.pretrained_weight_path}")
            try:
                checkpoint = torch.load(self.pretrained_weight_path, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

                keys_to_delete = [k for k in state_dict.keys() if "relative_position" in k]
                for k in keys_to_delete:
                    del state_dict[k]

                self.swin.load_state_dict(state_dict, strict=False)
                print("权重加载成功（已自适应 M=8 窗口维度）！")
            except Exception as e:
                print(f"权重加载失败：{str(e)}，使用随机初始化！")
        else:
            print("未找到本地权重文件，使用随机初始化！")

    def forward(self, x):
        swin_feats = self.swin(x)

        enhanced_feats = []
        for i, (feat, edge_extractor) in enumerate(zip(swin_feats, self.edge_extractors)):
            B, C, H, W = feat.shape
            feat_1d = feat.view(B, C, -1).transpose(1, 2)

            edge_bias = edge_extractor(feat_1d, H, W)
            edge_bias = edge_bias.transpose(1, 2).view(B, 1, H, W)

            enhanced_feat = feat + feat * edge_bias
            enhanced_feats.append(enhanced_feat)

        se_feats = [self.se_layers[i](f) for i, f in enumerate(enhanced_feats)]

        return {
            "res2": se_feats[0],
            "res3": se_feats[1],
            "res4": se_feats[2],
            "res5": se_feats[3]
        }

    def output_shape(self):
        return {
            "res2": ShapeSpec(channels=self.out_channels[0], stride=4),
            "res3": ShapeSpec(channels=self.out_channels[1], stride=8),
            "res4": ShapeSpec(channels=self.out_channels[2], stride=16),
            "res5": ShapeSpec(channels=self.out_channels[3], stride=32),
        }


if __name__ == "__main__":
    from detectron2.config import get_cfg

    cfg = get_cfg()

    test_input = torch.randn(2, 3, 500, 530)
    backbone = SwinBackbone(cfg, ShapeSpec(channels=3))

    with torch.no_grad():
        features = backbone(test_input)

    print("=" * 60)
    print("LeafSeg-T Edge-Aware Swin 骨干测试：")
    print("=" * 60)
    for name, tensor in features.items():
        print(f"层级: {name} | 尺寸: {tensor.shape} | 包含SE校准: True")