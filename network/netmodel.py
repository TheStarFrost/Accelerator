# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math


class NeuroImagingNet(nn.Module):
    """端到端神经影像重建网络"""

    def __init__(self):
        super().__init__()

        # 时序编码模块
        self.temporal_encoder = TemporalEncoder()

        # 空间解码模块
        self.spatial_decoder = SpatialDecoder()

        self._init_weights()

    def _init_weights(self):
        """Xavier初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """输入输出维度保持128x128"""
        # 时序特征提取 [B, 128, 128] → [B, 2048]
        features = self.temporal_encoder(x)

        # 空间重建 [B, 2048] → [B, 128, 128]
        return self.spatial_decoder(features)


class TemporalEncoder(nn.Module):
    """时序特征提取器"""

    def __init__(self):
        super().__init__()

        # 位置编码生成器
        self.pos_encoder = PositionalEncoder()

        # 特征扩展网络
        self.feature_mlp = FeatureExpander(
            input_dim=192,  # 128+64
            output_dim=2048,
            hidden_dims=[512, 1024, 2048],
        )

        # 时序聚合层
        self.temporal_pool = TemporalAggregator()

    def forward(self, x):
        """时序处理流程"""
        # 添加位置编码 [B, 128, 128] → [B, 128, 192]
        x = self.pos_encoder(x)

        # 特征扩展 [B, 128, 192] → [B, 128, 2048]
        x = self.feature_mlp(x)

        # 时序聚合 [B, 128, 2048] → [B, 2048]
        return self.temporal_pool(x)


class PositionalEncoder(nn.Module):
    """时序位置编码器"""

    def __init__(self, seq_len=128, feat_dim=128, d_model=64):
        super().__init__()
        self.register_buffer(
            "pe", self._gen_positional_encoding(seq_len, d_model)
        )
        self.projection = nn.Linear(feat_dim + d_model, feat_dim + d_model)

    def _gen_positional_encoding(self, seq_len, d_model):
        """生成正弦位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [128, 64]

    def forward(self, x):
        """特征与位置编码拼接,补足探测器信息"""
        pe = self.pe.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, 128, 64]
        return torch.cat([x, pe], dim=-1)  # [B, 128, 192]


class FeatureExpander(nn.Module):
    """特征维度扩展网络"""

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            ]
            current_dim = h_dim
        self.net = nn.Sequential(*layers[:-1])  # 移除最后一层的Dropout
        self.final_proj = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        """维度变换处理"""
        B, T, D = x.shape
        x = x.view(-1, D)  # [B*T, D]
        x = self.net(x)
        x = self.final_proj(x)
        return x.view(B, T, -1)  # 恢复时序维度


class TemporalAggregator(nn.Module):
    """时序特征聚合器"""

    def __init__(self, input_dim=2048, hidden_dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        self.post_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """注意力加权池化"""
        attn_weights = self.attention(x)  # [B, T, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
        return self.dropout(self.post_norm(pooled))


class SpatialDecoder(nn.Module):
    """空间重建解码器"""

    def __init__(self, input_dim=2048, output_dim=1):
        super().__init__()

        # 潜在空间到特征图
        self.latent_proj = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 256),  # 8*8*128
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2),
        )

        # 渐进式上采样模块
        self.upscale_blocks = nn.Sequential(
            DecoderBlock(256, 128, scale=2),  # 4→8
            DecoderBlock(128, 64, scale=2),  # 8→16
            DecoderBlock(64, 32, scale=2),  # 16→32
            DecoderBlock(32, 16, scale=2),  # 32→64
            DecoderBlock(16, 8, scale=2),  # 64→128
            OutputProjection(8, output_dim),  # 最终投影
        )

    def forward(self, x):
        """潜在向量到空间重建"""
        x = self.latent_proj(x)
        x = x.view(-1, 256, 4, 4)  # [B, C, H, W]
        x = self.upscale_blocks(x)  # 输出 [B, 1, 128, 128]
        return x.squeeze(1)  # 移除通道 → [B, 128, 128]


class DecoderBlock(nn.Module):
    """解码器基础模块"""

    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=False
        )
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(self.upsample(x))


class OutputProjection(nn.Module):
    """最终输出投影"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, 3), nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # 验证网络结构
    test_input = torch.randn(4, 128, 128)
    model = NeuroImagingNet()

    # 前向传播测试
    output = model(test_input)
    print(f"输入维度: {test_input.shape}")  # [4, 128, 128]
    print(f"输出维度: {output.shape}")  # [4, 128, 128]

    # 梯度回传测试
    output.mean().backward()
    print("梯度流验证通过")

    # 中间特征维度验证
    features = model.temporal_encoder(test_input)
    print(f"聚合特征维度: {features.shape}")  # [4, 2048]
