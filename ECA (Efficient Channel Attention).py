# 在 YOLOv5 或 YOLOv8 中接入 ECA 模块
class ECA(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size // 2, groups=channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))

# 类似 CBAM，我们可以在 YOLOv5 的 Backbone 或 Neck 部分插入 ECA
