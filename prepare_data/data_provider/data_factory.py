# Copyright 2024 The Time-LLM Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*- Factory Design Pattern for Data Loaders
# Đây là một "factory" (nhà máy) để tạo ra các đối tượng tải dữ liệu (DataLoader).
# Dựa trên tên của bộ dữ liệu được cung cấp trong tham số đầu vào (`args.data`),
# file này sẽ chọn đúng lớp `Dataset` tương ứng (ví dụ: `Dataset_ETT_hour` cho 'ETTh1')
# từ file `data_loader.py`.
#
# Chức năng chính:
# 1. Ánh xạ tên bộ dữ liệu (chuỗi) tới các lớp Dataset cụ thể thông qua `data_dict`.
# 2. Cấu hình các tham số cho DataLoader như kích thước batch (batch_size), có xáo trộn (shuffle) hay không,
#    và số lượng tiến trình con (num_workers).
# 3. Khởi tạo và trả về một đối tượng `Dataset` và một đối tượng `DataLoader` đã được cấu hình sẵn sàng
#    để sử dụng trong vòng lặp huấn luyện hoặc đánh giá.

from prepare_data.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'NetworksKPI': Dataset_Custom, # Thêm bộ dữ liệu NetworksKPI
    'm4': Dataset_M4,
    'TelecomKPI': Dataset_Custom
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader