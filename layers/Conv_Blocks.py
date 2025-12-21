"""
Tệp này định nghĩa các khối tích chập (Convolutional Blocks) lấy cảm hứng từ kiến trúc Inception,
thường được sử dụng trong các mô hình thị giác máy tính và đã được điều chỉnh cho các nhiệm vụ chuỗi thời gian.

Mục đích chính của các khối này là để trích xuất đặc trưng (feature extraction) ở nhiều quy mô (multi-scale)
khác nhau một cách đồng thời. Điều này đạt được bằng cách áp dụng nhiều bộ lọc tích chập với các kích thước
kernel khác nhau lên cùng một dữ liệu đầu vào và sau đó kết hợp các kết quả đầu ra.

- Inception_Block_V1: Sử dụng các kernel 2D đối xứng với các kích thước khác nhau (ví dụ: 1x1, 3x3, 5x5...)
  để nắm bắt các mẫu ở các độ dài khác nhau.
- Inception_Block_V2: Sử dụng các kernel 2D bất đối xứng (ví dụ: 1x3 và 3x1) để giảm số lượng tham số
  và chi phí tính toán so với các kernel đối xứng lớn, trong khi vẫn có khả năng trích xuất đặc trưng hiệu quả.

Các khối này có thể được dùng như một phần của bộ mã hóa (encoder) trong các mô hình dự báo chuỗi thời gian dựa trên CNN.
"""
import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
