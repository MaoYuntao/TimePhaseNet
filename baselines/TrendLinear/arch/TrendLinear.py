import torch
import torch.nn as nn

class SeriesDecomposition(nn.Module):
    """序列分解模块"""

    def __init__(self, kernel_size):
        super().__init__()
        # 使用平均池化模拟移动平均
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            count_include_pad=False  # 排除填充值的影响
        )

    def forward(self, x):
        # 输入维度: [B, L, N]
        x_trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]
        x_seasonal = x - x_trend # [B, L, N]
        return x_trend, x_seasonal # [B, L, N]


class TrendLinear(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.seq_len = model_args["seq_len"]
        self.pred_len = model_args["pred_len"]

        # 序列分解配置
        self.decomposition = SeriesDecomposition(kernel_size=13)

        # 双分支处理结构
        self.trend_net = self._build_subnet()
        self.seasonal_net = self._build_subnet()

        # 自适应融合参数
        self.alpha = nn.Parameter(torch.ones(1))
        # 初始化参数
        self._init_weights()

    def _build_subnet(self):
        """构建趋势/季节子网络"""
        return nn.Sequential(
            nn.Linear(self.seq_len, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(0.2),
            nn.Linear(64, self.pred_len)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 手动设置适合LeakyReLU的初始化增益
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # 减小偏置初始值

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        # 输入处理 [B, L, N, 3] -> [B, L, N]
        x = history_data[..., 0]  # 使用目标特征 #[B, L, N]

        # 序列分解
        trend, seasonal = self.decomposition(x.permute(0, 2, 1))  # 分解需要[C, L]格式
        trend = trend.permute(0, 2, 1)  # 恢复[B, L, N]
        seasonal = seasonal.permute(0, 2, 1)#[B, L, N]

        # 标准化处理
        seq_last = x[:, -1:, :].detach() #[B, 1, N]
        trend = (trend - seq_last).permute(0, 2, 1)  # [B, N, L]
        seasonal = (seasonal - seq_last).permute(0, 2, 1)# [B, N, L]

        # 双分支预测
        trend_pred = self.trend_net(trend).permute(0, 2, 1)  # [B, pred_len, N]
        seasonal_pred = self.seasonal_net(seasonal).permute(0, 2, 1)
        # 自适应融合
        prediction = self.alpha * trend_pred + (1 - self.alpha) * seasonal_pred + seq_last
        #prediction = (1 - self.alpha) * seasonal_pred + seq_last
        #prediction = 0.8 * trend_pred + 0.2 * seasonal_pred + seq_last

        return prediction.unsqueeze(-1)  # [B, pred_len, N, 1]



