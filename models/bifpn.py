import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """标准的 3x3 深度可分离卷积，对应论文中的 DWConv_{3x3}"""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FusionNode(nn.Module):
    """
    对应论文公式 (5):
    F_fuse(X, Y) = Sigmoid(alpha)*X + Sigmoid(beta)*Y + gamma*DWConv_3x3(X)
    """

    def __init__(self, channels):
        super().__init__()
        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

        # 对应公式的深度可分离卷积，用于强化 X (通常是当前尺度的本征特征) 的局部细节
        self.dwconv = DepthwiseSeparableConv(channels)

        # 融合后的标准 3x3 卷积，用于特征平滑 (参照 Fig 8 标注的 3x3 Conv)
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        # 公式 (5) 的具体实现
        w_x = torch.sigmoid(self.alpha)
        w_y = torch.sigmoid(self.beta)

        fused = w_x * x + w_y * y + self.gamma * self.dwconv(x)
        return self.post_conv(fused)


class BiFPNLayer(nn.Module):
    """单层 BiFPN 的具体拓扑实现，对应论文 Figure 8"""

    def __init__(self, channels):
        super().__init__()

        # Top-down (自顶向下) 融合节点
        self.td_fuse4 = FusionNode(channels)
        self.td_fuse3 = FusionNode(channels)
        self.td_fuse2 = FusionNode(channels)

        # Bottom-up (自底向上) 融合节点
        self.bu_fuse3 = FusionNode(channels)
        self.bu_fuse4 = FusionNode(channels)
        self.bu_fuse5 = FusionNode(channels)

        # 参照 Fig 8 的 3x3/2 Maxpool 下采样
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        P2, P3, P4, P5 = features

        # ================= Step 1: Top-Down Pathway =================
        P5_td = P5

        P5_up = F.interpolate(P5_td, scale_factor=2, mode="nearest")
        P4_td = self.td_fuse4(P4, P5_up)

        P4_up = F.interpolate(P4_td, scale_factor=2, mode="nearest")
        P3_td = self.td_fuse3(P3, P4_up)

        P3_up = F.interpolate(P3_td, scale_factor=2, mode="nearest")
        P2_td = self.td_fuse2(P2, P3_up)

        # ================= Step 2: Bottom-Up Pathway =================
        F2_out = P2_td

        # Fig 8 中间节点接收 3 个输入：原始输入、TD 特征、下采样特征。
        # 为严格符合公式 (5) 的双输入结构，这里将原始特征与 TD 特征相加作为主特征 X
        F2_down = self.max_pool(F2_out)
        F3_out = self.bu_fuse3(P3_td + P3, F2_down)

        F3_down = self.max_pool(F3_out)
        F4_out = self.bu_fuse4(P4_td + P4, F3_down)

        F4_down = self.max_pool(F4_out)
        F5_out = self.bu_fuse5(P5_td + P5, F4_down)

        return [F2_out, F3_out, F4_out, F5_out]


class BiFPN(nn.Module):
    """完整的 BiFPN 模块，包含通道对齐与公式 (6) 的 L=2 层堆叠"""

    def __init__(self, in_channels, out_channels=256, num_layers=2):
        super().__init__()
        self.out_channels = out_channels

        # 1. 输入通道调整 (1x1 Conv)，将骨干特征对齐到统一的维度
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])

        # 2. 堆叠 BiFPN 层 (论文指定 L=2)
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels) for _ in range(num_layers)
        ])

        # 3. 对应公式 (6) 的层间跨接缩放系数 gamma
        self.layer_gammas = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(num_layers - 1)
        ])

    def forward(self, features):
        # 初始化提取特征 (对应论文公式 6 中的 \tilde{P}_i)
        base_feats = [self.lateral_convs[i](f) for i, f in enumerate(features)]

        out_feats = base_feats
        for i, layer in enumerate(self.bifpn_layers):
            if i == 0:
                out_feats = layer(out_feats)
            else:
                gamma = self.layer_gammas[i - 1]
                input_feats = [base + gamma * prev_out for base, prev_out in zip(base_feats, out_feats)]
                out_feats = layer(input_feats)

        # 注：根据论文架构，CBAM 应在此模块输出后调用，所以此处直接返回多尺度特征
        return out_feats


# 测试代码
if __name__ == "__main__":
    # 模拟经过 SE 通道注意力增强后的骨干输出特征层
    test_feats = [
        torch.randn(2, 96, 128, 128),  # P2 (stride 4)
        torch.randn(2, 192, 64, 64),  # P3 (stride 8)
        torch.randn(2, 384, 32, 32),  # P4 (stride 16)
        torch.randn(2, 768, 16, 16)  # P5 (stride 32)
    ]

    # 根据论文设定，构建 L=2 的 BiFPN
    bifpn = BiFPN(in_channels=[96, 192, 384, 768], out_channels=256, num_layers=2)
    output_feats = bifpn(test_feats)

    print("=" * 60)
    print("🌿 LeafSeg-T 改进型 BiFPN 测试结果 (严格对齐 Eq.5 & Eq.6)：")
    print("=" * 60)
    for i, f in enumerate(output_feats):
        print(f"输出 Level {i + 2} (F{i + 2}): 形状 -> {f.shape}")