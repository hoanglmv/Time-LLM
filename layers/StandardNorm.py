"""
Tệp này triển khai một lớp chuẩn hóa chuyên dụng cho chuỗi thời gian được gọi là
**RevIN (Reversible Instance Normalization)**.

Mục đích chính của RevIN là để giải quyết vấn đề "thay đổi phân phối" (distribution shift) thường gặp trong
dữ liệu chuỗi thời gian, nơi các đặc tính thống kê (như trung bình, phương sai) của dữ liệu thay đổi theo thời gian.

Cơ chế này hoạt động theo hai giai đoạn:
1.  **Chuẩn hóa ('norm' mode)**:
    - Trước khi đưa vào mô hình chính, mỗi chuỗi dữ liệu trong một batch sẽ được chuẩn hóa độc lập
      bằng cách trừ đi trung bình và chia cho độ lệch chuẩn của chính nó.
    - Các giá trị thống kê (trung bình và độ lệch chuẩn) này được lưu lại cho từng chuỗi.

2.  **Phi chuẩn hóa ('denorm' mode)**:
    - Sau khi mô hình đưa ra dự báo (trên dữ liệu đã chuẩn hóa), lớp RevIN sẽ sử dụng các giá trị thống kê
      đã lưu trước đó để biến đổi ngược kết quả dự báo trở về thang đo (scale) ban đầu.

Quá trình "thuận nghịch" này giúp mô hình ổn định hơn trong quá trình huấn luyện và không bị ảnh hưởng bởi sự
thay đổi về độ lớn (magnitude) hay xu hướng (trend) của các chuỗi thời gian khác nhau.
"""
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
