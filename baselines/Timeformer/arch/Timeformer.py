import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer, FlashAttention, FlowAttention
from .Embed import DataEmbedding_inverted
import numpy as np
from basicts.utils import data_transformation_4_xformer


class MultiScaleFeatureExtractor(nn.Module):

    def __init__(self, in_dim, out_dim, scales=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        for s in scales:
            self.convs.append(
                nn.Conv1d(in_dim, out_dim // len(scales), kernel_size=s, padding=s // 2)
            )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Linear(out_dim // len(scales) * len(scales), out_dim)

    def forward(self, x):
        B, L, D = x.shape
        x = x.permute(0, 2, 1)  # [B, D, L]

        features = []
        for conv in self.convs:
            f = conv(x)  # [B, D//3, L]
            f = self.adaptive_pool(f).squeeze(-1)  # [B, D//3]
            features.append(f)

        fused = torch.cat(features, dim=-1)  # [B, D]
        return self.fusion(fused).unsqueeze(1)  # [B, 1, D]


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    def __init__(self, feature_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.GELU(),
            nn.Linear(feature_size // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        # features: list of [B, L, D]
        stacked = torch.stack(features, dim=1)  # [B, N, L, D]
        B, N, L, D = stacked.shape
        attn_weights = self.attention(stacked.view(B * N, L, D)).view(B, N, L, 1)
        fused = (stacked * attn_weights).sum(dim=1)
        return fused


class SeriesDecomposition(nn.Module):
    """序列分解模块"""

    def __init__(self, kernel_size):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, L, D]
        x_trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_trend, x_seasonal


class Timeformer(nn.Module):
    def __init__(self, **model_args):
        super(self).__init__()
        # 初始化参数
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.output_attention = model_args['output_attention']
        self.enc_in = model_args['enc_in']
        self.dec_in = model_args['dec_in']
        self.c_out = model_args['c_out']
        self.factor = model_args["factor"]
        self.d_model = model_args['d_model']
        self.n_heads = model_args['n_heads']
        self.d_ff = model_args['d_ff']
        self.embed = model_args['embed']
        self.freq = model_args["freq"]
        self.dropout = model_args["dropout"]
        self.activation = model_args['activation']
        self.e_layers = model_args['e_layers']
        self.d_layers = model_args['d_layers']
        self.use_norm = model_args['use_norm']

        # 新增模块初始化
        self.decomposition = SeriesDecomposition(kernel_size=25)
        self.multiscale_extractor = MultiScaleFeatureExtractor(self.enc_in, self.d_model)
        self.feature_fusion = AdaptiveFeatureFusion(self.d_model)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                       output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # 改进的投影层
        self.projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, self.pred_len)
        )

        # 辅助预测头
        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.d_model, self.pred_len // 2),
            nn.GELU(),
            nn.Linear(self.pred_len // 2, self.pred_len)
        )

    def forward_xformer(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                        x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                        enc_self_mask: torch.Tensor = None,
                        dec_self_mask: torch.Tensor = None,
                        dec_enc_mask: torch.Tensor = None) -> torch.Tensor:

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 序列分解
        x_trend, x_seasonal = self.decomposition(x_enc)

        # 多尺度特征提取
        global_features = self.multiscale_extractor(x_enc)

        # Embedding
        enc = self.enc_embedding(x_seasonal, x_mark_enc)  # 对季节性成分进行嵌入

        # 特征融合
        enc_out = torch.cat([enc, global_features.expand(-1, enc.shape[1], -1)], dim=-1)
        enc_out = self.feature_fusion([enc_out[:, :, :self.d_model], enc_out[:, :, self.d_model:]])

        # Encoding
        enc_out_residual = enc_out.clone()
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out += enc_out_residual  # 残差连接

        # 多任务预测
        main_pred = self.projector(enc_out).permute(0, 2, 1)[:, :, :self.enc_in]
        aux_pred = self.auxiliary_head(enc_out).permute(0, 2, 1)[:, :, :self.enc_in]

        # 结果融合
        dec_out = main_pred + 0.3 * aux_pred

        # 加入趋势成分
        dec_out += x_trend[:, -self.pred_len:, :]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:

        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(
            history_data=history_data,
            future_data=future_data,
            start_token_len=0
        )
        prediction = self.forward_xformer(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec
        )
        return prediction.unsqueeze(-1)